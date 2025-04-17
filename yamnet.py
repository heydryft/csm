import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])  # Limit to 2GB
    except RuntimeError as e:
        print(e)

import tensorflow_hub as hub
import numpy as np

model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

@tf.function
def predict(audio_input):
    return model(audio_input)

def has_speech(audio_data):
    scores, embeddings, log_mel_spectrogram = predict(audio_data)
    mean_scores = scores.numpy().mean(axis=0)
    speech_score = mean_scores[0]
    return speech_score > 0.5