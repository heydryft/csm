import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import ssl
import sounddevice as sd
import queue

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load YAMNet model
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

SAMPLE_RATE = 16000  # Hz

# Queue for audio chunks
q = queue.Queue()
recording = []

# Background audio callback
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# Thread to collect audio until Enter is pressed
def record_until_enter():
    global recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        print("Recording... Press Enter to stop.")
        input()
        print("Stopping...")
        while not q.empty():
            recording.append(q.get())

# Start recording in main thread
record_until_enter()

# Concatenate and convert to numpy array
audio_data = np.concatenate(recording, axis=0).flatten()

# Run YAMNet
scores, embeddings, log_mel_spectrogram = model(audio_data)

# Load class names
def class_names_from_csv(class_map_csv_text):
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (_, _, display_name) in csv.reader(class_map_csv)]
    return class_names[1:]  # Skip CSV header

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

# Print detected events
mean_scores = scores.numpy().mean(axis=0)
print("\nDetected Events (score > 0.1):")
for i, score in enumerate(mean_scores):
    if score > 0.1:
        print(f"{class_names[i]}: {score:.3f}")