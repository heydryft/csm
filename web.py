from tts_orpheus import OrpheusModel
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import struct
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import colorama
from colorama import Fore, Back
colorama.init()
import llm
import whisper
import torch
from datetime import datetime
from fastapi.responses import FileResponse

whisper_model = whisper.load_model("turbo", device="cuda")

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

# Audio buffer for each client
audio_buffers = {}
# Partial transcription buffer for each client
partial_transcriptions = {}
# Flag to track if we're waiting for more audio for a complete utterance
waiting_for_complete_utterance = {}
# Minimum silence duration to consider an utterance complete (in seconds)
UTTERANCE_END_SILENCE = 0.5
# Minimum audio duration to attempt transcription (in seconds)
MIN_TRANSCRIPTION_DURATION = 0.3
# Maximum buffer size before forcing transcription (in seconds)
MAX_BUFFER_DURATION = 5.0
# Last timestamp when audio was received for each client
last_audio_timestamp = {}

# Speech streaming connections
speech_connections = {}

current_client_stream = {}

# Silence timers for each client
silence_timers = {}
SILENCE_TIMEOUT = 15  # 5 seconds

# Initialize transcription model if available
if TRANSCRIPTION_AVAILABLE:
    print(f"{Fore.CYAN}Initializing transcription model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.CYAN}Using device: {device}")
    # sense_voice_model = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=True)

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
    silence_message = f"<User has been silent for {SILENCE_TIMEOUT} seconds, take the conversation ahead ask questions, do not question their silence>"
    
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

def reset_silence_timer(client_id: str):
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

async def process_audio_chunk(audio_chunk: bytes, client_id: str):
    """Process an audio chunk from the WebSocket and add it to the transcription queue"""
    # Reset the silence timer whenever we receive audio
    reset_silence_timer(client_id)
    
    # Convert audio chunk to numpy array
    samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Initialize buffers for this client if they don't exist
    if client_id not in audio_buffers:
        audio_buffers[client_id] = []
        partial_transcriptions[client_id] = ""
        waiting_for_complete_utterance[client_id] = False
        last_audio_timestamp[client_id] = time.time()
    
    # Record the timestamp of this audio chunk
    current_time = time.time()
    time_since_last_audio = current_time - last_audio_timestamp[client_id]
    last_audio_timestamp[client_id] = current_time
    
    # Add the new audio chunk to the buffer
    audio_buffers[client_id].append(samples)
    
    # Calculate total buffered audio duration
    total_samples = sum(len(chunk) for chunk in audio_buffers[client_id])
    buffer_duration = total_samples / 16000  # Assuming 16kHz sample rate
    
    # Check if we should process the buffer
    should_process = False
    is_complete_utterance = False
    
    # If there's a significant silence after speech, consider it a complete utterance
    if time_since_last_audio > UTTERANCE_END_SILENCE and buffer_duration > MIN_TRANSCRIPTION_DURATION:
        should_process = True
        is_complete_utterance = True
        debug(f"[DEBUG] Complete utterance detected after {time_since_last_audio:.2f}s silence")
    
    # If the buffer is getting too large, process it anyway
    elif buffer_duration > MAX_BUFFER_DURATION:
        should_process = True
        debug(f"[DEBUG] Processing buffer due to max duration reached: {buffer_duration:.2f}s")
    
    # Process the buffer if needed
    if should_process:
        # Concatenate all audio chunks
        combined_audio = np.concatenate(audio_buffers[client_id])
        
        # Create transcription queue if it doesn't exist
        if client_id not in transcription_queues:
            transcription_queues[client_id] = asyncio.Queue()
            # Start processing task
            asyncio.create_task(process_transcription_queue(client_id))
        
        # Add to transcription queue with complete utterance flag
        await transcription_queues[client_id].put({
            "audio_array": combined_audio,
            "timestamp": current_time,
            "duration": buffer_duration,
            "is_complete_utterance": is_complete_utterance
        })
        
        # Clear the buffer if this was a complete utterance
        if is_complete_utterance:
            audio_buffers[client_id] = []
            waiting_for_complete_utterance[client_id] = False
        else:
            # Keep half of the buffer for context in next transcription
            half_samples = total_samples // 2
            samples_to_keep = 0
            new_buffer = []
            
            for i in range(len(audio_buffers[client_id]) - 1, -1, -1):
                chunk = audio_buffers[client_id][i]
                if samples_to_keep + len(chunk) <= half_samples:
                    new_buffer.insert(0, chunk)
                    samples_to_keep += len(chunk)
                else:
                    # Add partial chunk if needed
                    if samples_to_keep < half_samples:
                        samples_remaining = half_samples - samples_to_keep
                        new_buffer.insert(0, chunk[-samples_remaining:])
                    break
            
            audio_buffers[client_id] = new_buffer

async def process_transcription_queue(client_id: str):
    """Process audio chunks from the transcription queue and generate transcripts"""
    try:
        queue = transcription_queues[client_id]
        while True:
            try:
                queue_wait_start = debug(f"[DEBUG] Waiting for next item in transcription queue")
                task = await queue.get()
                debug(f"[DEBUG] Got item from transcription queue", queue_wait_start)
                
                # Process the audio segment
                audio_array = task["audio_array"]
                timestamp = task["timestamp"]
                duration = task.get("duration", 0)
                is_complete_utterance = task.get("is_complete_utterance", False)
                
                # Skip processing if the audio is too short
                if duration and duration < MIN_TRANSCRIPTION_DURATION:
                    debug(f"[DEBUG] Skipping short audio segment: {duration}s")
                    queue.task_done()
                    continue
                
                # Check if transcription is available
                if not TRANSCRIPTION_AVAILABLE:
                    debug(f"[DEBUG] Transcription is not available. Skipping.")
                    queue.task_done()
                    continue
                
                # Transcribe the audio
                transcript = None
                try:
                    transcription_start = debug(f"[DEBUG] Starting transcription of audio segment")
                    audio_tensor = torch.tensor(audio_array)
                    
                    # Transcribe with Whisper
                    result = whisper_model.transcribe(audio_tensor, fp16=False, language="en")
                    transcript = result["text"].strip()
                    
                    # Apply heuristics to determine if this is a partial transcription
                    is_partial = False
                    
                    # Check if transcript ends with incomplete sentence
                    if not is_complete_utterance:
                        # Common sentence-ending punctuation
                        sentence_endings = ['.', '!', '?', '"', ')', ']']
                        # Check if transcript ends with a complete sentence
                        if transcript and transcript[-1] not in sentence_endings:
                            is_partial = True
                        
                        # Check if transcript ends with filler words that suggest continuation
                        filler_endings = ["um", "uh", "like", "so", "and", "but", "or", "because", "if", "when"]
                        for filler in filler_endings:
                            if transcript.lower().endswith(filler):
                                is_partial = True
                                break
                    
                    debug(f"[DEBUG] Transcription complete: '{transcript}'")
                    debug(f"[DEBUG] Is partial: {is_partial}, Is complete utterance: {is_complete_utterance}")
                    
                    # Store partial transcription
                    if is_partial:
                        partial_transcriptions[client_id] = transcript
                        waiting_for_complete_utterance[client_id] = True
                        
                        # Send partial transcript to client for display, but don't send to LLM yet
                        await manager.send_text(json.dumps({
                            "type": "partial_transcript",
                            "text": transcript,
                            "timestamp": timestamp
                        }), client_id)
                        
                        debug(f"[DEBUG] Sent partial transcript to client, waiting for complete utterance")
                        queue.task_done()
                        continue
                    
                    # If we were waiting for a complete utterance and now have one
                    if waiting_for_complete_utterance[client_id]:
                        waiting_for_complete_utterance[client_id] = False
                        debug(f"[DEBUG] Complete utterance received after waiting")
                    
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

    current_client_stream[client_id] = stream_id
    
    executor = ThreadPoolExecutor()

    executor.submit(stream_speech, prompt, "dan", stream_id, client_id, asyncio.get_event_loop())

def stream_speech(prompt: str, voice: str, stream_id: str, client_id: str = None, loop: asyncio.AbstractEventLoop = None):
    """Generate speech and put audio chunks into the queue"""
    global audio_stream_idx
    current_stream_idx = audio_stream_idx
    audio_stream_idx += 1

    stream_start = time.monotonic()
    time_to_first_byte = None
    total_frames = 0

    syn_tokens = model.generate_speech(prompt=prompt, voice=voice, max_tokens=2000, request_id=stream_id)

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

    for audio_chunk in syn_tokens:
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
                        asyncio.to_thread(reset_silence_timer, client_id)
                    else:
                        model.stop_stream(stream_id)
                        should_stop = True  # Set flag to stop processing
                        break
                except Exception as e:
                    debug(f"[ERROR] Failed to send audio chunk to speech WebSocket: {str(e)}")

            if time_to_first_byte is None:
                time_to_first_byte = time.monotonic() - stream_start

    duration = total_frames / 24000
    end_time = time.monotonic()
    elapsed = end_time - stream_start
    debug(f"[METRICS] [Stream {current_stream_idx}] Audio stream completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds, time to first byte: {time_to_first_byte:.2f} seconds", stream_start)

# Add debug utility function with timing
def debug(message, start_time=None):
    """Debug function with optional timing information
    
    Args:
        message: The debug message to print
        start_time: Optional start time for timing calculations. If provided, elapsed time will be shown.
    """
    current_time = time.time()
    timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    if start_time is not None:
        elapsed = current_time - start_time
        formatted_message = f"[{timestamp}] {message} (elapsed: {elapsed:.4f}s)"
    else:
        formatted_message = f"[{timestamp}] {message}"
    
    print(formatted_message)
    return current_time  # Return current time so it can be used as start_time for subsequent calls

@app.websocket("/ws/tts/{client_id}")
async def websocket_tts(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for TTS streaming"""
    connect_start = debug(f"[DEBUG] TTS WebSocket connecting for client {client_id}")
    await manager.connect(websocket, client_id)
    debug(f"[DEBUG] TTS WebSocket connected for client {client_id}", connect_start)
    try:
        # Start the silence timer when the connection is established
        reset_silence_timer(client_id)
        
        while True:
            try:
                # Receive message from the client
                receive_start = debug(f"[DEBUG] Waiting for message from client {client_id}")
                message = await websocket.receive()
                debug(f"[DEBUG] Received message from client {client_id}", receive_start)
                
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
                        elif msg_type == "vad_end":
                            # Reset the silence timer when VAD ends
                            reset_silence_timer(client_id)
                        
                    except json.JSONDecodeError:
                        await manager.send_text(json.dumps({"error": "Invalid JSON"}), client_id)
                    except Exception as e:
                        await manager.send_text(json.dumps({"error": str(e)}), client_id)
                
                # Handle binary messages (audio chunks)
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    process_start = debug(f"[DEBUG] Processing audio chunk from client {client_id}")
                    await process_audio_chunk(audio_chunk, client_id)
                    debug(f"[DEBUG] Processed audio chunk from client {client_id}", process_start)
                    
                    # Reset the silence timer when we receive audio
                    reset_silence_timer(client_id)
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
        greeting = await get_llm_response("<Greet>")
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