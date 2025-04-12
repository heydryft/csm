#!/usr/bin/env python3
import argparse
import collections
import queue
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
import ssl

import numpy as np

import pyaudio
import webrtcvad
import noisereduce as nr

import colorama
from colorama import Fore, Back
colorama.init()

ssl._create_default_https_context = ssl._create_unverified_context

# Optional dependencies for transcription
try:
    import torch
    from faster_whisper import WhisperModel
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    print("Faster-Whisper or PyTorch not found. Speech transcription will be disabled.")


def rms_energy(wave_bytes):
    samples = np.frombuffer(wave_bytes, dtype=np.int16)
    return np.sqrt(np.mean(samples.astype(np.float32)**2))


class AudioTranscriber:
    """
    A class for real-time audio capture, Voice Activity Detection (VAD),
    and optional speech transcription.
    """
    
    def __init__(self, 
                 sample_rate=16000, 
                 chunk_duration_ms=30, 
                 padding_duration_ms=300,
                 vad_aggressiveness=3,
                 silence_threshold_ms=500,
                 whisper_model="base",
                 output_file=None):
        """
        Initialize the AudioTranscriber with configuration parameters.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            padding_duration_ms: Padding duration before and after VAD in milliseconds
            vad_aggressiveness: VAD aggressiveness level (0-3)
            silence_threshold_ms: Silence duration to mark end of speech in milliseconds
            whisper_model: Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large')
            output_file: Path to save output, if None output is printed to console
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.silence_threshold_ms = silence_threshold_ms
        self.output_file = output_file
        
        # Calculate sizes based on parameters
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.padding_chunks = int(self.padding_duration_ms / self.chunk_duration_ms)
        
        # Set up Voice Activity Detector
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Set up queues for audio processing and speech segments
        self.audio_queue = queue.Queue()
        self.speech_segments_queue = queue.Queue()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Set up Whisper model
        print(f"{Fore.CYAN}Initializing Whisper model '{whisper_model}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{Fore.CYAN}Using device: {device}")
        self.whisper_model = WhisperModel(whisper_model, device=device)
        
        # Control flags
        self.running = False
        self.recording = False
        
        # Initialize buffers
        self.ring_buffer = collections.deque(maxlen=self.padding_chunks)
        self.voiced_frames = []
        self.speech_start_time = None
        self.silent_chunks_count = 0
        self.last_half_baked = None  # Store the last half-baked transcript
        self.last_partial_time = None  # Store the timestamp of the last partial transcript

    def is_half_baked(self, transcript):
        """
        Simple heuristic to determine if a transcript seems incomplete.
        For this example, if the transcript does not end with a typical sentence
        terminator (".", "!", or "?"), we consider it half baked.
        
        Args:
            transcript: The transcribed text string.
            
        Returns:
            True if the transcript appears incomplete, False otherwise.
        """
        # if it ends with ... its half baked
        if transcript and transcript[-3:] == '...':
            return True
        # if it doesn't end with ., !, or ? its half baked
        if transcript and transcript[-1] not in {'.', '!', '?'}:
            return True
        return False
    
    def record_audio(self):
        """
        Continuously record audio from the microphone and put it in the audio queue.
        This function runs in a separate thread.
        """
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
                # Read audio chunk from the microphone
                frame = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put((frame, datetime.now()))
                
        except Exception as e:
            print(f"{Fore.RED}Error in audio recording: {e}")
        finally:
            # Clean up
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
    
    def process_audio(self):
        """
        Process audio chunks from the queue, detect voice activity,
        and capture speech segments.
        This function runs in a separate thread.
        """
        try:
            while self.running:
                # Check if we need to timeout any partial transcripts
                self.check_partial_transcript_timeout()
                
                # Get audio chunk from the queue
                if self.audio_queue.empty():
                    time.sleep(0.01)  # Short sleep to prevent CPU hogging
                    continue
                
                frame, timestamp = self.audio_queue.get()
                
                # Check if frame is valid for VAD
                if len(frame) != self.chunk_size * 2:  # 16-bit = 2 bytes per sample
                    continue
                
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                
                # Always add to the ring buffer
                self.ring_buffer.append((frame, is_speech, timestamp))
                
                # Handle speech detection state machine
                if not self.recording:
                    # Start recording if speech is detected
                    if is_speech:
                        self.recording = True
                        self.speech_start_time = timestamp
                        
                        # Include the buffered frames that might contain the beginning of speech
                        self.voiced_frames = list(self.ring_buffer)
                        self.silent_chunks_count = 0
                else:
                    # Continue recording and track the speech/silence state
                    self.voiced_frames.append((frame, is_speech, timestamp))
                    
                    if not is_speech:
                        self.silent_chunks_count += 1
                        silent_duration_ms = self.silent_chunks_count * self.chunk_duration_ms
                        
                        if silent_duration_ms >= self.silence_threshold_ms:
                            # End of speech detected - process the segment
                            speech_end_time = timestamp
                            
                            self._process_speech_segment(
                                self.voiced_frames, 
                                self.speech_start_time, 
                                speech_end_time
                            )
                            
                            # Reset for next segment
                            self.recording = False
                            self.voiced_frames = []
                            self.speech_start_time = None
                            self.silent_chunks_count = 0
                    else:
                        # Reset silence counter when speech is detected again
                        self.silent_chunks_count = 0
                            
        except Exception as e:
            print(f"{Fore.RED}Error in audio processing: {e}")
    
    def check_partial_transcript_timeout(self):
        """
        Check if the last partial transcript has been waiting for more than 2 seconds.
        If so, print it as a full transcript and clear it.
        """
        if self.last_partial_time and self.last_half_baked:
            now = datetime.now()
            elapsed = (now - self.last_partial_time).total_seconds()
            if elapsed >= 2.0:  # 2 seconds timeout
                print(f"{Back.GREEN}TRANSCRIPT {Back.RESET}: {self.last_half_baked} [timeout]")
                # Clear the partial transcript
                self.last_half_baked = None
                self.last_partial_time = None
    
    def _process_speech_segment(self, frames, start_time, end_time):
        """
        Process a complete speech segment, optionally transcribe it,
        and add to the speech segments queue.
        
        Args:
            frames: List of tuples (audio_frame, is_speech, timestamp)
            start_time: Start time of the speech segment
            end_time: End time of the speech segment
        """
        # Extract audio data
        audio_data = b''.join([f[0] for f in frames])

        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Assume first 0.2s is mostly background noise
        noise_len = int(self.sample_rate * 0.2)
        noise_clip = samples[:noise_len]

        # Apply noise reduction
        reduced = nr.reduce_noise(y=samples, y_noise=noise_clip, sr=self.sample_rate)

        # Overwrite audio_data with denoised signal
        audio_data = reduced.astype(np.int16).tobytes() 

        energy = rms_energy(audio_data)
        if energy < 100:  # tweak this threshold as needed
            return
        
        # Calculate duration
        duration = (end_time - start_time).total_seconds()
        
        # Format timestamps for display
        start_str = start_time.strftime("%H:%M:%S.%f")[:-3]
        end_str = end_time.strftime("%H:%M:%S.%f")[:-3]
        
        transcript = None
        
        # Only process further if the segment is longer than a minimum threshold
        if duration < 0.1:  # Ignore very short segments
            return
        
        # Save audio to a temporary file for transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
        
        # Transcribe audio
        try:
            # Transcribe with Faster Whisper
            segments, info = self.whisper_model.transcribe(
                temp_filename,
                language="en",
                beam_size=1,             # greedy decoding
                best_of=1,               # no resampling
                temperature=0.0,          # deterministic decoding (greedy)
                condition_on_previous_text=False,  # lower latency per chunk
                initial_prompt=None,      # can be set to guide style
                no_speech_threshold=0.6,  # suppress low-confidence speech
                compression_ratio_threshold=2.4,  # prevents gibberish
            )
            
            # Collect all segments into a single transcript
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text)
            
            transcript = " ".join(transcript_parts).strip()

            # If we have a last half-baked transcript, join it with the current one
            if self.last_half_baked:
                transcript = f"{self.last_half_baked} {transcript}"
                self.last_half_baked = None  # Clear the last half-baked transcript
                self.last_partial_time = None  # Reset the partial transcript timer
        except Exception as e:
            print(f"{Fore.RED}Error in transcription: {e}")
            transcript = "[Transcription failed]"
        
        # Create a speech segment record
        segment = {
            "start_time": start_str,
            "end_time": end_str,
            "duration": duration,
            "audio_file": temp_filename,
            "transcript": transcript
        }
        
        # Add to the segments queue for potential further processing
        self.speech_segments_queue.put(segment)
        
        # Determine if the transcript appears complete or "half baked"
        if self.is_half_baked(transcript):
            # Store this half-baked transcript for the next segment
            self.last_half_baked = transcript
            # Store the timestamp when this partial transcript was created
            self.last_partial_time = datetime.now()
            # Inform the operator that a half baked response was logged
            print(f"{Back.YELLOW}PARTIAL {Back.RESET}{transcript}")
        else:
            # Print the transcript as final since it appears complete
            print(f"{Back.GREEN}TRANSCRIPT {Back.RESET}: {transcript}")
            # Reset the partial transcript timer
            self.last_partial_time = None
        
        # If desired, you can also output to a file if self.output_file is provided.
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(f"{datetime.now()} - Transcript from {start_str} to {end_str}\n")
                f.write(f"{transcript}\n\n")
    
    def start(self):
        """
        Start the audio transcription process.
        """
        self.running = True
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        try:
            # Keep the main thread alive to receive keyboard interrupts
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"{Fore.CYAN}\nStopping recording...")
            self.stop()
    
    def stop(self):
        """
        Stop the audio transcription process and clean up resources.
        """
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'record_thread') and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)
        
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # Clean up PyAudio
        self.audio.terminate()
        
        print("Recording stopped.")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Arguments namespace
    """
    parser = argparse.ArgumentParser(description="Real-Time Speech-to-Text System")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate in Hz (default: 16000)")
    parser.add_argument("--vad-aggressiveness", type=int, choices=[0, 1, 2, 3], default=3,
                        help="VAD aggressiveness (0-3): higher values filter more non-speech (default: 3)")
    parser.add_argument("--silence-threshold", type=int, default=500,
                        help="Silence duration to mark end of speech in ms (default: 500)")
    parser.add_argument("--enable-transcription", action="store_true",
                        help="Enable speech transcription with Whisper")
    parser.add_argument("--whisper-model", type=str, default="tiny",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: tiny)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save output (default: print to console)")
    
    return parser.parse_args()


def main():
    """
    Main entry point for the script.
    """
    args = parse_arguments()
    
    # Display configuration
    print("Real-Time Speech-to-Text System")
    print("-------------------------------")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"VAD aggressiveness: {args.vad_aggressiveness}")
    print(f"Silence threshold: {args.silence_threshold} ms")
    print(f"Whisper model: {args.whisper_model}")
    print(f"Output: {'File: ' + args.output_file if args.output_file else 'Console'}")
    print("-------------------------------")
    
    try:
        # Initialize and start the audio transcriber
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