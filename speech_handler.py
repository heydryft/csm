import requests
import pyaudio
import threading
import uuid
import io
import struct

# Track running audio streams using a dictionary
audio_streams = {}

def _play_streaming_audio(endpoint_url: str, stop_event: threading.Event):
    """
    Internal function to stream and play WAV audio correctly.
    Downloads the WAV stream, decodes it, and plays the raw PCM audio in real-time.
    """
    p = pyaudio.PyAudio()
    stream = None

    try:
        with requests.get(endpoint_url, stream=True) as response:
            if response.status_code == 200:
                print(f"[INFO] Streaming audio from {endpoint_url}...")
                
                # Variables to track WAV header processing
                header_processed = False
                buffer = io.BytesIO()
                
                # Stream setup variables
                nchannels = None
                framerate = None
                sampwidth = None
                data_offset = None
                
                for chunk in response.iter_content(chunk_size=4096):
                    if stop_event.is_set():
                        print(f"[INFO] Audio stream from {endpoint_url} stopped (early).")
                        return
                    
                    # Add chunk to buffer
                    buffer.write(chunk)
                    
                    # Process WAV header if not done yet
                    if not header_processed:
                        # Need at least 44 bytes for basic WAV header
                        if buffer.tell() >= 44:
                            buffer.seek(0)
                            
                            # Check if we have a valid WAV file
                            try:
                                # Read WAV header manually to find data chunk
                                if buffer.read(4) != b'RIFF':
                                    raise ValueError("Not a valid WAV file (RIFF header missing)")
                                    
                                # Skip file size
                                buffer.seek(8)
                                
                                # Check WAVE format
                                if buffer.read(4) != b'WAVE':
                                    raise ValueError("Not a valid WAV file (WAVE format missing)")
                                    
                                # Find fmt chunk
                                chunk_id = b''
                                while chunk_id != b'fmt ' and buffer.tell() < buffer.getbuffer().nbytes:
                                    chunk_id = buffer.read(4)
                                    if chunk_id == b'fmt ':
                                        # Read fmt chunk size
                                        fmt_size = struct.unpack('<I', buffer.read(4))[0]
                                        fmt_start = buffer.tell()
                                        
                                        # Read format data
                                        audio_format = struct.unpack('<H', buffer.read(2))[0]  # 1 for PCM
                                        nchannels = struct.unpack('<H', buffer.read(2))[0]
                                        framerate = struct.unpack('<I', buffer.read(4))[0]
                                        byte_rate = struct.unpack('<I', buffer.read(4))[0]
                                        block_align = struct.unpack('<H', buffer.read(2))[0]
                                        bits_per_sample = struct.unpack('<H', buffer.read(2))[0]
                                        sampwidth = bits_per_sample // 8
                                        
                                        # Skip to end of fmt chunk
                                        buffer.seek(fmt_start + fmt_size)
                                    else:
                                        # Skip other chunks
                                        chunk_size = struct.unpack('<I', buffer.read(4))[0]
                                        buffer.seek(buffer.tell() + chunk_size)
                                
                                # Find data chunk
                                chunk_id = b''
                                while chunk_id != b'data' and buffer.tell() < buffer.getbuffer().nbytes:
                                    chunk_id = buffer.read(4)
                                    if chunk_id == b'data':
                                        # Skip data chunk size
                                        buffer.read(4)
                                        data_offset = buffer.tell()
                                    else:
                                        # Skip other chunks
                                        chunk_size = struct.unpack('<I', buffer.read(4))[0]
                                        buffer.seek(buffer.tell() + chunk_size)
                                
                                if nchannels and framerate and sampwidth and data_offset:
                                    print(f"[INFO] WAV format: {nchannels}ch, {framerate}Hz, {sampwidth * 8}bit")
                                    
                                    # Create audio stream with correct parameters
                                    stream = p.open(format=p.get_format_from_width(sampwidth),
                                                    channels=nchannels,
                                                    rate=framerate,
                                                    output=True)
                                    
                                    # Mark header as processed
                                    header_processed = True
                                    
                                    # Extract and play audio data that we already have
                                    buffer.seek(data_offset)
                                    audio_data = buffer.read()
                                    if audio_data and not stop_event.is_set():
                                        stream.write(audio_data)
                                    
                                    # Clear buffer to save memory
                                    buffer = io.BytesIO()
                                else:
                                    # Incomplete header, wait for more data
                                    continue
                            except Exception as e:
                                print(f"[ERROR] Error processing WAV header: {e}")
                                return
                    else:
                        # Header already processed, play this chunk directly
                        if chunk and not stop_event.is_set() and stream:
                            stream.write(chunk)
                            # No need to keep in buffer since we're playing directly
                            buffer = io.BytesIO()
            else:
                print(f"[ERROR] Failed to fetch audio from {endpoint_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Exception while streaming from {endpoint_url}: {e}")
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()

def start_audio_stream(endpoint_url: str) -> str:
    """
    Starts streaming audio from the given endpoint URL.
    Returns a unique stream ID.
    """
    stream_id = str(uuid.uuid4())
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_play_streaming_audio,
        args=(endpoint_url, stop_event),
        daemon=True
    )
    audio_streams[stream_id] = stop_event
    thread.start()
    print(f"[INFO] Started stream {stream_id} from {endpoint_url}")
    return stream_id

def stop_audio_stream(stream_id: str):
    """
    Stops an audio stream using its stream ID.
    """
    if stream_id in audio_streams:
        audio_streams[stream_id].set()
        del audio_streams[stream_id]
        print(f"[INFO] Stream {stream_id} stopped.")
    else:
        print(f"[WARNING] No active stream with ID: {stream_id}")

def stop_all_audio_streams():
    """
    Stops all active audio streams.
    """
    for stream_id in list(audio_streams.keys()):
        stop_audio_stream(stream_id)

def speak(prompt: str, voice: str = "tara"):
    stream_id = start_audio_stream(f"http://localhost:8000/tts?prompt={prompt}&voice={voice}")
    return stream_id

def stop():
    stop_all_audio_streams()