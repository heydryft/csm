#!/usr/bin/env python3
import argparse
import collections
import queue
import sys
import threading
import time
from datetime import datetime
import ssl

import numpy as np
import pyaudio
import webrtcvad

import colorama
from colorama import Fore, Back
colorama.init()
import llm

import speech_handler

ssl._create_default_https_context = ssl._create_unverified_context

# Optional dependencies for transcription
try:
    import torch
    from omnisense.models import OmniSenseVoiceSmall
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    print("Faster-Whisper or PyTorch not found. Speech transcription will be disabled.")

def rms_energy(wave_bytes):
    samples = np.frombuffer(wave_bytes, dtype=np.int16)
    return np.sqrt(np.mean(samples.astype(np.float32)**2))

class AudioTranscriber:
    def __init__(self, sample_rate=16000, chunk_duration_ms=30, padding_duration_ms=300,
                 vad_aggressiveness=3, silence_threshold_ms=500,
                 whisper_model="base", output_file=None):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.silence_threshold_ms = silence_threshold_ms
        self.output_file = output_file

        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.padding_chunks = int(self.padding_duration_ms / self.chunk_duration_ms)

        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self.audio_queue = queue.Queue()
        self.speech_segments_queue = queue.Queue()
        self.transcription_queue = queue.Queue()

        self.audio = pyaudio.PyAudio()

        print(f"{Fore.CYAN}Initializing Whisper model '{whisper_model}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{Fore.CYAN}Using device: {device}")
        self.sense_voice_model = OmniSenseVoiceSmall("iic/SenseVoiceSmall")

        self.running = False
        self.recording = False

        self.ring_buffer = collections.deque(maxlen=self.padding_chunks)
        self.voiced_frames = []
        self.speech_start_time = None
        self.silent_chunks_count = 0
        self.last_half_baked = None
        self.last_partial_time = None

    def is_half_baked(self, transcript):
        return False
        if transcript and transcript[-3:] == '...':
            return True
        if transcript and transcript[-1] not in {'.', '!', '?'}:
            return True
        return False

    def record_audio(self):
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print(f"{Fore.CYAN}Recording started. Press Ctrl+C to stop.")

            while self.running:
                frame = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put((frame, datetime.now()))
        except Exception as e:
            print(f"{Fore.RED}Error in audio recording: {e}")
        finally:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()

    def process_audio(self):
        try:
            while self.running:
                self.check_partial_transcript_timeout()

                if self.audio_queue.empty():
                    time.sleep(0.01)
                    continue

                frame, timestamp = self.audio_queue.get()

                if len(frame) != self.chunk_size * 2:
                    continue

                is_speech = self.vad.is_speech(frame, self.sample_rate)
                self.ring_buffer.append((frame, is_speech, timestamp))

                if not self.recording:
                    if is_speech:
                        speech_handler.stop()
                        self.recording = True
                        self.speech_start_time = timestamp
                        self.voiced_frames = list(self.ring_buffer)
                        self.silent_chunks_count = 0
                else:
                    self.voiced_frames.append((frame, is_speech, timestamp))

                    if not is_speech:
                        self.silent_chunks_count += 1
                        silent_duration_ms = self.silent_chunks_count * self.chunk_duration_ms
                        if silent_duration_ms >= self.silence_threshold_ms:
                            speech_end_time = timestamp
                            self._process_speech_segment(
                                self.voiced_frames,
                                self.speech_start_time,
                                speech_end_time
                            )
                            self.recording = False
                            self.voiced_frames = []
                            self.speech_start_time = None
                            self.silent_chunks_count = 0
                    else:
                        self.silent_chunks_count = 0
        except Exception as e:
            print(f"{Fore.RED}Error in audio processing: {e}")

    def check_partial_transcript_timeout(self):
        if self.last_partial_time and self.last_half_baked:
            now = datetime.now()
            elapsed = (now - self.last_partial_time).total_seconds()
            if elapsed >= 2.0:
                print(f"{Back.GREEN}TRANSCRIPT {Back.RESET}: {self.last_half_baked} [timeout]")
                self.last_half_baked = None
                self.last_partial_time = None

    def _process_speech_segment(self, frames, start_time, end_time):
        raw_audio = b''.join([f[0] for f in frames])
        samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        if rms_energy(raw_audio) < 150:
            return

        duration = (end_time - start_time).total_seconds()
        if duration < 0.1:
            return

        start_str = start_time.strftime("%H:%M:%S.%f")[:-3]
        end_str = end_time.strftime("%H:%M:%S.%f")[:-3]

        self.transcription_queue.put({
            "audio_array": samples,
            "start_time": start_time,
            "end_time": end_time,
            "start_str": start_str,
            "end_str": end_str,
            "duration": duration
        })

    def process_transcription_queue(self):
        try:
            while self.running:
                if self.transcription_queue.empty():
                    time.sleep(0.01)
                    continue

                task = self.transcription_queue.get()
                audio_array = task["audio_array"]
                start_str = task["start_str"]
                end_str = task["end_str"]
                duration = task["duration"]

                transcript = None
                try:
                    segments = self.sense_voice_model.transcribe(
                        audio_array,
                        language="en",
                        batch_size=256,
                        progressbar=False,
                        textnorm="withitn"
                    )
                    transcript_parts = [f"<{seg.event} ({seg.emotion})> {seg.text} </{seg.event}>" for seg in segments]
                    transcript = " ".join(transcript_parts).strip()

                    if self.last_half_baked:
                        transcript = f"{self.last_half_baked} {transcript}"
                        self.last_half_baked = None
                        self.last_partial_time = None

                except Exception as e:
                    print(f"{Fore.RED}Error in transcription: {e}")
                    transcript = "[Transcription failed]"

                segment = {
                    "start_time": start_str,
                    "end_time": end_str,
                    "duration": duration,
                    "transcript": transcript,
                }

                self.speech_segments_queue.put(segment)

                if self.is_half_baked(transcript):
                    self.last_half_baked = transcript
                    self.last_partial_time = datetime.now()
                    print(f"{Back.YELLOW}PARTIAL{Back.RESET}{transcript}")
                else:
                    print(f"{Back.GREEN}TRANSCRIPT{Back.RESET}: {transcript}")
                    reply = llm.respond(transcript)
                    speech_handler.speak(reply)
                    print(f"{Back.BLUE}MUSE{Back.RESET}: {reply}")
                    self.last_partial_time = None

                if self.output_file:
                    with open(self.output_file, 'a') as f:
                        f.write(f"{datetime.now()} - Transcript from {start_str} to {end_str}\n")
                        f.write(f"{transcript}\n\n")

                self.transcription_queue.task_done()
        except Exception as e:
            print(f"{Fore.RED}Error in transcription processing: {e}")

    def start(self):
        self.running = True
        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.record_thread.start()

        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.process_thread.start()

        self.transcription_thread = threading.Thread(target=self.process_transcription_queue, daemon=True)
        self.transcription_thread.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"{Fore.CYAN}\nStopping recording...")
            self.stop()

    def stop(self):
        self.running = False
        if hasattr(self, 'record_thread') and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        if hasattr(self, 'transcription_thread') and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1.0)
        self.audio.terminate()
        print("Recording stopped.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Speech-to-Text System")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--vad-aggressiveness", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--silence-threshold", type=int, default=500)
    parser.add_argument("--enable-transcription", action="store_true")
    parser.add_argument("--whisper-model", type=str, default="tiny",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--output-file", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("Real-Time Speech-to-Text System")
    print("-------------------------------")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"VAD aggressiveness: {args.vad_aggressiveness}")
    print(f"Silence threshold: {args.silence_threshold} ms")
    print(f"Whisper model: {args.whisper_model}")
    print(f"Output: {'File: ' + args.output_file if args.output_file else 'Console'}")
    print("-------------------------------")

    try:
        transcriber = AudioTranscriber(
            sample_rate=args.sample_rate,
            vad_aggressiveness=args.vad_aggressiveness,
            silence_threshold_ms=args.silence_threshold,
            whisper_model=args.whisper_model,
            output_file=args.output_file
        )
        transcriber.start()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
