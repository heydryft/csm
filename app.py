from tts_orpheus import OrpheusModel
import wave
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

prompts = [
    "Lately, I've been thinking about how much our routines shape the way we feel every day. Like, even small things—waking up a bit earlier, having a proper breakfast, or just stepping outside for a short walk—can make such a difference.",
    "Also, I had this random moment yesterday where I started a conversation with a stranger at the coffee shop, and it turned out to be one of the nicest chats I've had in a while. It reminded me how disconnected we sometimes get, especially when life gets busy or stressful.",
]

def generate_and_save(prompt: str, index: int, voice: str):
    print(f"Starting generation for prompt {index}")
    start_time = time.monotonic()

    syn_tokens = model.generate_speech(prompt=prompt, voice=voice)

    output_path = f"output_{index}.wav"

    time_to_first_byte = None

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
            if time_to_first_byte is None:
                time_to_first_byte = time.monotonic() - start_time

    duration = total_frames / 24000
    end_time = time.monotonic()
    elapsed = end_time - start_time
    print(f"Prompt {index} completed in {elapsed:.2f} seconds, audio duration: {duration:.2f} seconds, time to first byte: {time_to_first_byte:.2f} seconds")
    return index, elapsed, duration

def main():
    global model
    model = OrpheusModel(model_name="./Orpheus-3b-AWQ", tokenizer="./Orpheus-3b-AWQ")

    start_all = time.monotonic()
    results = []
    print("Starting generation for prompts")
    voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_and_save, prompt, voices[idx % len(voices)], idx) for idx, prompt in enumerate(prompts)]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.monotonic() - start_all
    print("\nSummary:")
    for idx, elapsed, duration in sorted(results):
        print(f"Prompt {idx}: Generation Time = {elapsed:.2f}s, Audio Duration = {duration:.2f}s")
    print(f"Total time for all: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
