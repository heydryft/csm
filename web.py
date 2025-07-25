from kokoro import KPipeline
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import struct
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import colorama
from colorama import Fore, Back
colorama.init()
import llm
from fastapi.responses import FileResponse
import webrtcvad
# import yamnet

import whisper
from utils import debug

import base64

# model_size = "distil-large-v3"
model_size = "tiny"

# Load whisper model
whisper_model = whisper.load_model(model_size)

vad = webrtcvad.Vad(3)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Kokoro TTS pipeline
model = KPipeline(lang_code='a')

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
SILENCE_TIMEOUT = 10  # 5 seconds

PREPEND_LAST_TRANSCRIPTION_TIMEOUT = 3.0

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
    start_time = debug(f"[DEBUG] Silence timeout for client {client_id}")

    global silence_idx
    silence_idx += 1
    
    # Generate a silence notification message
    silence_message = f"<User has been silent for {SILENCE_TIMEOUT} seconds, ask if they're still there in the call>"
    
    # Get response from LLM for the silence
    try:
        llm_start = debug(f"[DEBUG] Getting LLM response for silence")
        reply = await get_llm_response(silence_message)
        debug(f"{Back.BLUE}SILENCE RESPONSE{Back.RESET}: {reply}", llm_start)
        
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
        debug(f"[ERROR] Error handling silence timeout: {str(e)}")

async def start_silence_timer(client_id: str, delay: float = SILENCE_TIMEOUT):
    """Start or reset the silence timer for a client"""
    # Cancel existing timer if there is one
    cancel_silence_timer(client_id)
    
    # Create a new timer
    loop = asyncio.get_event_loop()
    async def silence_timer_task(client_id: str, delay: float):
        await asyncio.sleep(delay)
        return client_id

    silence_timers[client_id] = loop.create_task(
        silence_timer_task(client_id, delay)
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

# Global variables for audio chunk buffering
audio_buffer = {}
audio_buffer_timestamps = {}
audio_processing_tasks = {}

async def process_audio_chunk(audio_chunk: bytes, client_id: str):
    """Process an audio chunk from the WebSocket and add it to the transcription queue"""
    
    # Convert audio chunk to numpy array
    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

    # Initialize buffer for this client if it doesn't exist
    if client_id not in audio_buffer:
        audio_buffer[client_id] = []
        audio_buffer_timestamps[client_id] = time.time()
    
    # Add the audio chunk to the buffer
    audio_buffer[client_id].append(samples)
    audio_buffer_timestamps[client_id] = time.time()
    
    # Cancel any existing processing task
    if client_id in audio_processing_tasks and not audio_processing_tasks[client_id].done():
        audio_processing_tasks[client_id].cancel()
    
    # Schedule a new processing task after 0.2 seconds of silence
    audio_processing_tasks[client_id] = asyncio.create_task(
        process_buffered_audio_after_silence(client_id, 0.2)
    )

async def process_buffered_audio_after_silence(client_id: str, silence_duration: float):
    """Process buffered audio after a period of silence"""
    try:
        # Wait for the silence duration
        await asyncio.sleep(silence_duration)
        
        # Check if we have any buffered audio
        if client_id in audio_buffer and audio_buffer[client_id]:
            # Combine all buffered audio chunks
            combined_samples = np.concatenate(audio_buffer[client_id])
            total_duration = len(combined_samples) / 16000
            
            # Initialize transcription queue if needed
            if client_id not in transcription_queues:
                transcription_queues[client_id] = asyncio.Queue()
                # Start processing task
                asyncio.create_task(process_transcription_queue(client_id))
            
            # Add the combined audio to the transcription queue
            await transcription_queues[client_id].put({
                "audio_array": combined_samples,
                "timestamp": audio_buffer_timestamps[client_id],
                "duration": total_duration
            })
            
            debug(f"[DEBUG] Processed buffered audio after {silence_duration}s silence: {total_duration}s")
            
            # Clear the buffer after processing
            audio_buffer[client_id] = []
    except asyncio.CancelledError:
        # Task was cancelled because new audio arrived
        pass
    except Exception as e:
        debug(f"{Fore.RED}Error processing buffered audio: {e}")

last_transcription_time = {}
last_transcription_text = {}

async def process_transcription_queue(client_id: str):
    """Process audio chunks from the transcription queue and generate transcripts"""
    try:
        queue = transcription_queues[client_id]
        while True:
            try:
                task = await queue.get()
                debug(f"[DEBUG] Got item from transcription queue")
                
                # Process the audio segment
                audio_array = task["audio_array"]
                timestamp = task["timestamp"]
                audio_duration = len(audio_array) / 16000

                if audio_duration < 0.1:
                    debug(f"[DEBUG] Skipping short audio segment: {audio_duration}s")
                    continue
                
                # Transcribe the audio
                transcript = None

                # Time check
                now = time.time()
                prepend = False

                # Should we prepend previous transcript?
                if client_id in last_transcription_time:
                    time_diff = now - last_transcription_time[client_id]
                    if time_diff < PREPEND_LAST_TRANSCRIPTION_TIMEOUT:
                        prepend = True

                # Transcription
                try:
                    async def check_speech(audio_data):
                        # yamnet_start = debug(f"[DEBUG] Checking for speech in audio segment")
                        # if not yamnet.has_speech(audio_array):
                        #     debug(f"[DEBUG] Skipping non-speech audio segment", yamnet_start)
                        #     return False
                        # debug(f"[DEBUG] Speech detected", yamnet_start)
                        return True

                    async def transcribe_audio(audio_data):
                        transcription_start = debug(f"[DEBUG] Starting transcription of audio segment")
                        # Standard whisper API is different from faster_whisper
                        # Convert audio to float32 in range [-1, 1]
                        if audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        # Whisper expects mono audio at 16kHz
                        result = whisper_model.transcribe(audio_data, language="en")
                        debug(f"[DEBUG] Transcription complete", transcription_start)
                        return result

                    has_speech, result = await asyncio.gather(check_speech(audio_array), transcribe_audio(audio_array))
                    if not has_speech:
                        continue

                    # Standard whisper returns a dict with 'text' key
                    transcript = result['text']

                    if prepend and client_id in last_transcription_text:
                        transcript = last_transcription_text[client_id].strip() + " " + transcript.strip()

                    # Store this as last transcription
                    last_transcription_text[client_id] = transcript
                    last_transcription_time[client_id] = now

                    debug(f"[DEBUG] Final transcript (possibly prepended): {transcript}")

                except Exception as e:
                    debug(f"{Fore.RED}Error in transcription: {e}")
                    transcript = "[Transcription failed]"
                
                if transcript:
                    
                    # Get response from LLM
                    llm_start = debug(f"[DEBUG] Getting LLM response for transcript")
                    reply = await get_llm_response(transcript)
                    debug(f"{Back.BLUE}RESPONSE{Back.RESET}: {reply}", llm_start)

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
                debug(f"{Fore.RED}Error in transcription: {e}")
                await manager.send_text(json.dumps({"error": f"Transcription failed: {str(e)}"}), client_id)
            
            queue.task_done()
    except Exception as e:
        debug(f"{Fore.RED}Error in transcription processing: {e}")

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

async def get_llm_response(transcript):
    """Get a response from the LLM"""
    # Use the existing llm.respond function
    return await llm.respond(transcript)

async def stream_audio_websocket(prompt: str, voice: str, client_id: str):
    """Generate speech and stream it to the client"""
    if not prompt:
        await manager.send_text(json.dumps({"error": "Missing prompt"}), client_id)
        return
    
    # Generate a unique ID for this stream
    stream_id = f"{int(time.time())}_{voice}_{hash(prompt) % 10000}"

    current_client_stream[client_id] = stream_id
    
    # For Kokoro, use these voice options: 'af_heart', 'en_us_amy', 'en_us_andy', 'en_us_ashley', etc.
    # Default to 'af_heart' if voice is not specified or not compatible
    kokoro_voice = 'af_heart'
    if voice in ['af_heart', 'en_us_amy', 'en_us_andy', 'en_us_ashley']:
        kokoro_voice = voice
    else:
        debug(f"[INFO] Voice '{voice}' not found in Kokoro, using default 'af_heart'")
    
    executor = ThreadPoolExecutor()

    executor.submit(stream_speech, prompt, kokoro_voice, stream_id, client_id, asyncio.get_event_loop())

def stream_speech(prompt: str, voice: str, stream_id: str, client_id: str = None, loop: asyncio.AbstractEventLoop = None):
    """Generate speech and put audio chunks into the queue"""
    global audio_stream_idx
    current_stream_idx = audio_stream_idx
    audio_stream_idx += 1

    stream_start = time.monotonic()
    time_to_first_byte = None
    total_frames = 0

    # Generate speech using Kokoro
    try:
        generator = model(prompt, voice=voice)
    except Exception as e:
        debug(f"[ERROR] Failed to generate speech with Kokoro: {str(e)}")
        return

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
                debug(f"[ERROR] Failed to send audio chunk to speech WebSocket: {str(e)}")

    cancel_silence_timer(client_id)
    try:
        for i, (gs, ps, audio_data) in enumerate(generator):
            # Convert audio data to bytes (Kokoro uses 24kHz sample rate)
            # Convert PyTorch tensor to NumPy array first
            audio_data_np = audio_data.cpu().numpy() if hasattr(audio_data, 'cpu') else audio_data
            audio_chunk = (audio_data_np * 32767).astype(np.int16).tobytes()
            
            if should_stop:  # Check if we should stop processing
                break
                    
            frame_count = len(audio_chunk) // (2 * 1)  # 16-bit mono
            total_frames += frame_count
            
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
                    except Exception as e:
                        debug(f"[ERROR] Failed to send audio chunk to speech WebSocket: {str(e)}")
        
        # End of speech notification
        if client_id and client_id in speech_connections and speech_connections[client_id]:
            for ws in speech_connections[client_id]:
                try:
                    if stream_id == current_client_stream[client_id]:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps({"type": "tts_end", "stream_id": stream_id})),
                            loop
                        )
                except Exception as e:
                    debug(f"[ERROR] Failed to send end notification to speech WebSocket: {str(e)}")
    except Exception as e:
        debug(f"[ERROR] Error in Kokoro speech generation: {str(e)}")

    if time_to_first_byte is None and total_frames > 0:
        time_to_first_byte = time.monotonic() - stream_start

    duration = total_frames / 24000

    end_time = time.monotonic()
    elapsed = end_time - stream_start

    asyncio.run_coroutine_threadsafe(start_silence_timer(client_id, (duration - elapsed) + SILENCE_TIMEOUT), loop)

    debug(f"[METRICS] [Stream {current_stream_idx}] Audio stream completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds, time to first byte: {time_to_first_byte:.2f} seconds", stream_start)

@app.websocket("/ws/tts/{client_id}")
async def websocket_tts(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for TTS streaming"""
    connect_start = debug(f"[DEBUG] TTS WebSocket connecting for client {client_id}")
    await manager.connect(websocket, client_id)
    debug(f"[DEBUG] TTS WebSocket connected for client {client_id}", connect_start)
    try:
        while True:
            try:
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
                            tts_start = debug(f"[DEBUG] Processing TTS request for client {client_id}")
                            await stream_audio_websocket(prompt, voice, client_id)
                            debug(f"[DEBUG] TTS request processed for client {client_id}", tts_start)
                        
                    except json.JSONDecodeError:
                        await manager.send_text(json.dumps({"error": "Invalid JSON"}), client_id)
                    except Exception as e:
                        await manager.send_text(json.dumps({"error": str(e)}), client_id)
                
                # Handle binary messages (audio chunks)
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    await process_audio_chunk(audio_chunk, client_id)

                elif "vad_start" in message:
                    await cancel_silence_timer(client_id)

            except RuntimeError as e:
                # Handle "Cannot call 'receive' once a disconnect message has been received"
                if "disconnect message" in str(e):
                    debug(f"[DEBUG] Client {client_id} disconnected")
                    break
                else:
                    raise
    
    except WebSocketDisconnect:
        debug(f"[DEBUG] WebSocket disconnected for client {client_id}")
    except Exception as e:
        debug(f"[ERROR] WebSocket error: {str(e)}")
    finally:
        # Always clean up resources when the connection ends
        debug(f"[DEBUG] Cleaning up resources for client {client_id}")
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
        connect_start = debug(f"[DEBUG] Speech WebSocket connecting for client {client_id}")
        await websocket.accept()
        debug(f"[DEBUG] Speech WebSocket connected for client {client_id}", connect_start)
        
        # Store the connection for later use
        if client_id not in speech_connections:
            speech_connections[client_id] = []
        speech_connections[client_id].append(websocket)
        
        # Send a greeting message to the client
        greeting = await get_llm_response("<Greet the user on the call>")
        await websocket.send_text(json.dumps({"type": "response", "message": greeting}))
        await stream_audio_websocket(greeting, "tara", client_id)
        
        # Keep the connection open
        while True:
            try:
                # Just wait for messages to keep the connection alive
                message = await websocket.receive()
                # If we get a close message, break the loop
                if message.get("type") == "websocket.disconnect":
                    break
            except Exception as e:
                if "disconnect" in str(e).lower():
                    break
                debug(f"[ERROR] Speech WebSocket error: {str(e)}")
                break
    except WebSocketDisconnect:
        debug(f"[DEBUG] Speech WebSocket disconnected for client {client_id}")
    except Exception as e:
        debug(f"[ERROR] Speech WebSocket error: {str(e)}")
    finally:
        # Remove this connection from the list
        if client_id in speech_connections and websocket in speech_connections[client_id]:
            speech_connections[client_id].remove(websocket)
            if not speech_connections[client_id]:
                del speech_connections[client_id]
        debug(f"[DEBUG] Speech WebSocket cleaned up for client {client_id}")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)