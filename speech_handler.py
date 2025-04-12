import requests
import pyaudio
import threading
import uuid

# Track running audio streams using a dictionary
audio_streams = {}

def _play_streaming_audio(endpoint_url: str, stop_event: threading.Event,
                          rate=16000, channels=1, format=pyaudio.paInt16):
    """
    Internal function to stream and play audio until stopped.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    output=True)

    try:
        with requests.get(endpoint_url, stream=True) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=4096):
                    if stop_event.is_set():
                        print(f"[INFO] Audio stream from {endpoint_url} stopped.")
                        break
                    if chunk:
                        stream.write(chunk)
            else:
                print(f"[ERROR] Failed to fetch audio from {endpoint_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Error while streaming from {endpoint_url}: {e}")
    finally:
        stream.stop_stream()
        stream.close()
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