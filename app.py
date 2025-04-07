from tts_orpheus import OrpheusModel
import wave
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

prompts = [
    "Hi, how you doin?",
    "Welcome to the Orpheus TTS demo!",
    "The quick brown fox jumps over the lazy dog.",
    "This is a parallel processing test.",
    "Python concurrency is powerful!",
]

def generate_and_save(prompt: str, index: int):
    print(f"Starting generation for prompt {index}")
    start_time = time.monotonic()

    syn_tokens = model.generate_speech(prompt=prompt, voice="tara")

    output_path = f"output_{index}.wav"
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

    duration = total_frames / 24000
    end_time = time.monotonic()
    elapsed = end_time - start_time
    print(f"Prompt {index} completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds")
    return index, elapsed, duration

def main():
    global model
    model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

    start_all = time.monotonic()
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_and_save, prompt, idx) for idx, prompt in enumerate(prompts)]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.monotonic() - start_all
    print("\nSummary:")
    for idx, elapsed, duration in sorted(results):
        print(f"Prompt {idx}: Generation Time = {elapsed:.2f}s, Audio Duration = {duration:.2f}s")
    print(f"Total time for all: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
