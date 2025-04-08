import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from tts_orpheus import OrpheusModel
import struct
import asyncio
import os
import wave

app = FastAPI()
model = OrpheusModel(model_name="heydryft/Orpheus-3b-FT-AWQ", tokenizer="heydryft/Orpheus-3b-FT-AWQ")

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )

audio_stream_idx = 0

async def stream_audio(prompt: str, voice: str):
    global audio_stream_idx
    current_stream_idx = audio_stream_idx
    audio_stream_idx += 1

    start_time = time.monotonic()
    yield create_wav_header()
    yield b'\0' * 2

    total_frames = 0
    time_to_first_byte = None
    buffer_size = 4096  # Larger buffer size for more efficient streaming
    audio_buffer = bytearray()

    with wave.open(f"output_{current_stream_idx}.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        for chunk in model.generate_speech(prompt=prompt, voice=voice, max_tokens=8192):
            if chunk is None:
                break
            
            # Add to buffer
            audio_buffer.extend(chunk)
            
            # Only yield when buffer reaches threshold for more efficient streaming
            if len(audio_buffer) >= buffer_size:
                if time_to_first_byte is None:
                    time_to_first_byte = time.monotonic() - start_time
                yield bytes(audio_buffer)
                wf.writeframes(audio_buffer)
                frame_count = len(audio_buffer) // 2
                total_frames += frame_count
                audio_buffer = bytearray()
        
        # Send any remaining audio in the buffer
        if audio_buffer:
            yield bytes(audio_buffer)
            total_frames += len(audio_buffer) // 2
            wf.writeframes(audio_buffer)

    duration = total_frames / 24000
    end_time = time.monotonic()
    elapsed = end_time - start_time
    print(f"[METRICS] [Stream {current_stream_idx}] Audio stream completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds, time to first byte: {time_to_first_byte:.2f} seconds")


@app.get("/tts")
async def tts(prompt: str, voice: str = "tara"):
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    return StreamingResponse(stream_audio(prompt, voice), media_type="audio/wav")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
