from tts_orpheus import OrpheusModel
import wave
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import struct
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = OrpheusModel(model_name="heydryft/Orpheus-3b-FT-AWQ", tokenizer="heydryft/Orpheus-3b-FT-AWQ")

audio_stream_idx = 0

# Queue to store audio chunks for streaming
audio_queues = {}

def stream_speech(prompt: str, voice: str, stream_id: str):
    """Generate speech and put audio chunks into the queue"""
    global audio_stream_idx
    current_stream_idx = audio_stream_idx
    audio_stream_idx += 1

    start_time = time.monotonic()
    
    # Create the queue if it doesn't exist
    if stream_id not in audio_queues:
        audio_queues[stream_id] = asyncio.Queue()
    
    # Add WAV header to the queue
    audio_queues[stream_id].put_nowait(create_wav_header())
    
    syn_tokens = model.generate_speech(prompt=prompt, voice=voice, max_tokens=8192)

    time_to_first_byte = None
    total_frames = 0
    
    with wave.open(f"output_{stream_id}.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (2 * 1)  # 16-bit mono
            total_frames += frame_count
            
            # Add the chunk to the queue
            audio_queues[stream_id].put_nowait(audio_chunk)
            wf.writeframes(audio_chunk)

            if time_to_first_byte is None:
                time_to_first_byte = time.monotonic() - start_time

    duration = total_frames / 24000
    end_time = time.monotonic()
    elapsed = end_time - start_time
    print(f"[METRICS] [Stream {current_stream_idx}] Audio stream completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds, time to first byte: {time_to_first_byte:.2f} seconds")
    
    # Signal the end of the stream
    audio_queues[stream_id].put_nowait(None)


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1) -> bytes:
    """Create a WAV header for streaming audio"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0  # For streaming, we don't know the final size
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


async def stream_audio_generator(stream_id: str) -> AsyncGenerator[bytes, None]:
    """Async generator that yields audio chunks from the queue"""
    if stream_id not in audio_queues:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    queue = audio_queues[stream_id]
    
    while True:
        # Wait for the next chunk
        chunk = await queue.get()
        
        # If None is received, the stream is complete
        if chunk is None:
            break

        yield chunk

    # Clean up the queue when done
    if stream_id in audio_queues:
        del audio_queues[stream_id]


@app.get("/tts")
async def tts(prompt: str, voice: str = "tara"):
    """TTS endpoint that returns a streaming response"""
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    
    # Generate a unique ID for this stream
    stream_id = f"{int(time.time())}_{voice}_{hash(prompt) % 10000}"
    print(f"[METRICS] Starting stream {stream_id}")
    
    executor = ThreadPoolExecutor()

    executor.submit(stream_speech, prompt, voice, stream_id)
    
    # Return a streaming response
    return StreamingResponse(
        stream_audio_generator(stream_id),
        media_type="audio/wav"
    )


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)