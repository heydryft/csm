from tts_orpheus import OrpheusModel
import wave
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import struct
import queue
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator, List, Dict
import json
import colorama
from colorama import Fore, Back
colorama.init()
import llm
import whisper
import torch

whisper_model = whisper.load_model("tiny", device="cuda")

def contains_words_from_tensor(audio_tensor: torch.Tensor, sample_rate: int) -> bool:
    """
    Transcribe from a waveform tensor and check for words.
    """
    # Resample if needed
    if sample_rate != 16000:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)

    # Whisper expects mono 16kHz NumPy array
    audio_np = audio_tensor.squeeze().numpy()
    result = whisper_model.transcribe(audio_np, fp16=False, language="en")

    text = result.get("text", "").strip()
    return len(text) > 0

# Optional dependencies for transcription
try:
    from omnisense.models import OmniSenseVoiceSmall
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    print("Faster-Whisper or PyTorch not found. Speech transcription will be disabled.")

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

# Queue for transcription processing
transcription_queues = {}

# Store speech segments and processing state
speech_segments = {}
recording_state = {}

# Speech streaming connections
speech_connections = {}

current_client_stream = {}

# Silence timers for each client
silence_timers = {}
SILENCE_TIMEOUT = 15  # 7 seconds

# Initialize transcription model if available
if TRANSCRIPTION_AVAILABLE:
    print(f"{Fore.CYAN}Initializing transcription model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.CYAN}Using device: {device}")
    sense_voice_model = OmniSenseVoiceSmall("iic/SenseVoiceSmall")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def send_text(self, message: str, client_id: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_text(message)
    
    async def send_bytes(self, data: bytes, client_id: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_bytes(data)

    async def broadcast(self, message: str):
        for client_id in self.active_connections:
            await self.send_text(message, client_id)

manager = ConnectionManager()

silence_idx = 0
async def handle_silence_timeout(client_id: str):
    """Handle when the user has been silent for the timeout period"""
    print(f"[DEBUG] Silence timeout for client {client_id}")

    global silence_idx
    silence_idx += 1
    
    # Generate a silence notification message
    silence_message = f"<Silence ({silence_idx})>"
    
    # Get response from LLM for the silence
    try:
        reply = await get_llm_response(silence_message)
        print(f"{Back.BLUE}SILENCE RESPONSE{Back.RESET}: {reply}")
        
        # Send the response to the client
        if client_id in speech_connections:
            await asyncio.gather(
                manager.send_text(json.dumps({
                    "type": "response",
                    "text": reply
                }), client_id),
                stream_audio_websocket(reply, "tara", client_id)
            )
    except Exception as e:
        print(f"[ERROR] Error handling silence timeout: {str(e)}")

def start_silence_timer(client_id: str):
    """Start or reset the silence timer for a client"""
    # Cancel existing timer if there is one
    cancel_silence_timer(client_id)
    
    # Create a new timer
    loop = asyncio.get_event_loop()
    silence_timers[client_id] = loop.create_task(
        asyncio.sleep(SILENCE_TIMEOUT, result=client_id)
    )
    
    # Set up the callback for when the timer completes
    silence_timers[client_id].add_done_callback(
        lambda task: asyncio.create_task(handle_silence_timeout(task.result()))
        if not task.cancelled() else None
    )

def cancel_silence_timer(client_id: str):
    """Cancel the silence timer for a client if it exists"""
    if client_id in silence_timers and silence_timers[client_id] is not None:
        silence_timers[client_id].cancel()
        silence_timers[client_id] = None

def rms_energy(wave_bytes):
    samples = np.frombuffer(wave_bytes, dtype=np.int16)
    return np.sqrt(np.mean(samples.astype(np.float32)**2))

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

async def process_transcription_queue(client_id: str):
    """Process audio chunks from the transcription queue and generate transcripts"""
    try:
        print(f"[DEBUG] Starting transcription queue processing for client {client_id}")
        queue = transcription_queues[client_id]
        while True:
            try:
                # Get the next item from the queue
                print(f"[DEBUG] Waiting for next item in transcription queue")
                task = await queue.get()
                print(f"[DEBUG] Got item from transcription queue")
                
                # Process the audio segment
                audio_array = task["audio_array"]
                timestamp = task["timestamp"]
                duration = task.get("duration", 0)
                
                # Skip processing if the audio is too short
                if duration and duration < 0.1:
                    print(f"[DEBUG] Skipping short audio segment: {duration}s")
                    queue.task_done()
                    continue
                
                # Check if transcription is available
                if not TRANSCRIPTION_AVAILABLE:
                    print(f"[DEBUG] Transcription is not available. Skipping.")
                    queue.task_done()
                    continue
                
                # Transcribe the audio
                transcript = None
                try:
                    print(f"[DEBUG] Starting transcription of audio segment")
                    audio_tensor = torch.tensor(audio_array)

                    # Run both tasks concurrently
                    # contains_words_task = contains_words_from_tensor(audio_tensor, 16000)
                    segments = sense_voice_model.transcribe(
                        audio_array,
                        language="en",
                        batch_size=256,
                        progressbar=False,
                        textnorm="withitn"
                    )

                    # if not contains_words:
                    #     queue.task_done()
                    #     continue
                    
                    print(f"[DEBUG] Transcription complete, got {len(segments)} segments")
                    transcript_parts = [f"<{seg.event} ({seg.emotion})> {seg.text} </{seg.event}>" for seg in segments]
                    transcript = " ".join(transcript_parts).strip()
                    print(f"[DEBUG] Final transcript: {transcript}")
                    
                except Exception as e:
                    print(f"{Fore.RED}Error in transcription: {e}")
                    transcript = "[Transcription failed]"
                
                if transcript:
                    
                    # Get response from LLM
                    reply = await get_llm_response(transcript)
                    print(f"{Back.BLUE}RESPONSE{Back.RESET}: {reply}")

                    await asyncio.gather(
                        manager.send_text(json.dumps({
                            "type": "transcript",
                            "text": transcript,
                            "timestamp": timestamp
                        }), client_id),

                        manager.send_text(json.dumps({
                            "type": "response",
                            "text": reply
                        }), client_id),

                        stream_audio_websocket(reply, "tara", client_id)
                    )
            except Exception as e:
                print(f"{Fore.RED}Error in transcription: {e}")
                await manager.send_text(json.dumps({"error": f"Transcription failed: {str(e)}"}), client_id)
            
            queue.task_done()
    except Exception as e:
        print(f"{Fore.RED}Error in transcription processing: {e}")

async def process_audio_chunk(audio_chunk: bytes, client_id: str):
    """Process an audio chunk from the WebSocket and add it to the transcription queue"""
    # Reset the silence timer whenever we receive audio
    start_silence_timer(client_id)
    
    # Convert audio chunk to numpy array
    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

    if client_id not in transcription_queues:
        transcription_queues[client_id] = asyncio.Queue()
        # Start processing task
        asyncio.create_task(process_transcription_queue(client_id))

    await transcription_queues[client_id].put({
        "audio_array": samples,
        "timestamp": time.time(),
        "duration": len(samples) / 16000
    })

async def process_speech_segment(frames, start_time, end_time, client_id):
    """Process a complete speech segment and add it to the transcription queue"""
    print(f"[DEBUG] Processing speech segment with {len(frames)} frames")
    # Combine all audio frames
    raw_audio = b''.join([f[0] for f in frames])
    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Check energy level again
    energy = rms_energy(raw_audio)
    print(f"[DEBUG] Combined audio energy: {energy}")
    if energy < 150:
        print(f"[DEBUG] Skipping segment due to low energy: {energy}")
        return
    
    # Calculate duration
    duration = end_time - start_time
    print(f"[DEBUG] Speech segment duration: {duration} seconds")
    if duration < 0.1:
        print(f"[DEBUG] Skipping segment due to short duration: {duration}")
        return
    
    # Create transcription queue if it doesn't exist
    if client_id not in transcription_queues:
        transcription_queues[client_id] = asyncio.Queue()
        # Start processing task
        print(f"[DEBUG] Created new transcription queue for client {client_id}")
        asyncio.create_task(process_transcription_queue(client_id))
    
    # Add to transcription queue
    print(f"[DEBUG] Adding speech segment to transcription queue")
    await transcription_queues[client_id].put({
        "audio_array": samples,
        "timestamp": time.time(),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration
    })
    print(f"[DEBUG] Added speech segment to queue successfully")

async def get_llm_response(transcript):
    """Get a response from the LLM"""
    # Use the existing llm.respond function
    return llm.respond(transcript)

async def stream_audio_websocket(prompt: str, voice: str, client_id: str):
    """Generate speech and stream it to the client"""
    if not prompt:
        await manager.send_text(json.dumps({"error": "Missing prompt"}), client_id)
        return
    
    # Generate a unique ID for this stream
    stream_id = f"{int(time.time())}_{voice}_{hash(prompt) % 10000}"
    print(f"[METRICS] Starting stream {stream_id}")

    current_client_stream[client_id] = stream_id
    
    executor = ThreadPoolExecutor()

    executor.submit(stream_speech, prompt, voice, stream_id, client_id, asyncio.get_event_loop())

def stream_speech(prompt: str, voice: str, stream_id: str, client_id: str = None, loop: asyncio.AbstractEventLoop = None):
    """Generate speech and put audio chunks into the queue"""
    global audio_stream_idx
    current_stream_idx = audio_stream_idx
    audio_stream_idx += 1

    start_time = time.monotonic()

    syn_tokens = model.generate_speech(prompt=prompt, voice=voice, max_tokens=2000, request_id=stream_id)

    time_to_first_byte = None
    total_frames = 0

    with wave.open(f"output_{stream_id}.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        should_stop = False  # Flag to indicate if we should stop processing
        
        # Send start message to client WebSocket
        if client_id and client_id in speech_connections and speech_connections[client_id]:
            for ws in speech_connections[client_id]:
                try:
                    if stream_id == current_client_stream[client_id]:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps({"type": "tts_start", "stream_id": stream_id})),
                            loop
                        )
                except Exception as e:
                    print(f"[ERROR] Failed to send audio chunk to speech WebSocket: {str(e)}")

        for audio_chunk in syn_tokens:
            if should_stop:  # Check if we should stop processing
                break
                
            frame_count = len(audio_chunk) // (2 * 1)  # 16-bit mono
            total_frames += frame_count

            wf.writeframes(audio_chunk)

            # Also send to speech WebSocket if client_id is provided
            if client_id and client_id in speech_connections and speech_connections[client_id]:
                # Send the audio chunk to all connected speech WebSockets for this client
                for ws in speech_connections[client_id]:
                    try:
                        if stream_id == current_client_stream[client_id]:
                            asyncio.run_coroutine_threadsafe(
                                ws.send_bytes(audio_chunk),
                                loop
                            )
                        else:
                            model.stop_stream(stream_id)
                            should_stop = True  # Set flag to stop processing
                            break
                    except Exception as e:
                        print(f"[ERROR] Failed to send audio chunk to speech WebSocket: {str(e)}")

            if time_to_first_byte is None:
                time_to_first_byte = time.monotonic() - start_time

    duration = total_frames / 24000
    end_time = time.monotonic()
    elapsed = end_time - start_time
    print(f"[METRICS] [Stream {current_stream_idx}] Audio stream completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds, time to first byte: {time_to_first_byte:.2f} seconds")


@app.websocket("/ws/tts/{client_id}")
async def websocket_tts(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for TTS streaming"""
    await manager.connect(websocket, client_id)
    try:
        # Start the silence timer when the connection is established
        start_silence_timer(client_id)
        
        while True:
            try:
                # Receive message from the client
                message = await websocket.receive()
                
                # Handle text messages
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type")

                        if msg_type == "tts_request":
                            prompt = data.get("prompt")
                            voice = data.get("voice", "tara")
                            
                            if not prompt:
                                await manager.send_text(json.dumps({"error": "Missing prompt"}), client_id)
                                continue
                            
                            # Process TTS and stream audio
                            await stream_audio_websocket(prompt, voice, client_id)
                        elif msg_type == "vad_end":
                            # Reset the silence timer when VAD ends
                            start_silence_timer(client_id)
                        
                    except json.JSONDecodeError:
                        await manager.send_text(json.dumps({"error": "Invalid JSON"}), client_id)
                    except Exception as e:
                        await manager.send_text(json.dumps({"error": str(e)}), client_id)
                
                # Handle binary messages (audio chunks)
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    await process_audio_chunk(audio_chunk, client_id)
            except RuntimeError as e:
                # Handle "Cannot call 'receive' once a disconnect message has been received"
                if "disconnect message" in str(e):
                    print(f"[DEBUG] Client {client_id} disconnected")
                    break
                else:
                    raise
    
    except WebSocketDisconnect:
        print(f"[DEBUG] WebSocket disconnected for client {client_id}")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {str(e)}")
    finally:
        # Always clean up resources when the connection ends
        print(f"[DEBUG] Cleaning up resources for client {client_id}")
        manager.disconnect(websocket, client_id)
        
        # Cancel the silence timer
        cancel_silence_timer(client_id)
        
        # Clean up queues
        if client_id in transcription_queues:
            del transcription_queues[client_id]
            
        # Clean up recording state
        if client_id in recording_state:
            del recording_state[client_id]
            
        # Clean up speech segments
        if client_id in speech_segments:
            del speech_segments[client_id]

@app.websocket("/ws/speech/{client_id}")
async def websocket_speech(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for streaming speech audio to the client"""
    try:
        await websocket.accept()
        print(f"[DEBUG] Speech WebSocket connected for client {client_id}")
        
        # Store the connection for later use
        if client_id not in speech_connections:
            speech_connections[client_id] = []
        speech_connections[client_id].append(websocket)
        
        # Send a greeting message to the client
        greeting = await get_llm_response("<Greet>")
        await websocket.send_text(json.dumps({"type": "response", "message": greeting}))
        await stream_audio_websocket(greeting, "tara", client_id)
        
        # Keep the connection open until client disconnects
        while True:
            try:
                # This is just to keep the connection alive and detect disconnections
                await websocket.receive_text()
            except Exception as e:
                if "disconnect" in str(e).lower():
                    break
                print(f"[ERROR] Speech WebSocket error: {str(e)}")
                break
    except WebSocketDisconnect:
        print(f"[DEBUG] Speech WebSocket disconnected for client {client_id}")
    except Exception as e:
        print(f"[ERROR] Speech WebSocket error: {str(e)}")
    finally:
        # Remove this connection from the list
        if client_id in speech_connections and websocket in speech_connections[client_id]:
            speech_connections[client_id].remove(websocket)
            if not speech_connections[client_id]:
                del speech_connections[client_id]
        print(f"[DEBUG] Speech WebSocket cleaned up for client {client_id}")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)