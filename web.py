from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from tts_orpheus import OrpheusModel
from concurrent.futures import ThreadPoolExecutor
import struct
import asyncio
from typing import Iterator

app = FastAPI()
model = OrpheusModel(model_name="./Orpheus-3b-AWQ", tokenizer="./Orpheus-3b-AWQ")
executor = ThreadPoolExecutor()

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

async def stream_audio(prompt: str, voice: str):
    yield create_wav_header()

    loop = asyncio.get_event_loop()

    def generate_chunks():
        for chunk in model.generate_speech(prompt=prompt, voice=voice, max_tokens=8192):
            yield chunk

    for chunk in await loop.run_in_executor(executor, lambda: generate_chunks()):
        yield chunk


@app.get("/tts")
async def tts(prompt: str, voice: str = "tara"):
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    return StreamingResponse(stream_audio(prompt, voice), media_type="audio/wav")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
