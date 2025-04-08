from tts_orpheus import OrpheusModel
import wave
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

prompts = [
    "Oh man... okay, so—there was this one time I, uh, totally bombed a group presentation 'cause I thought it was next week.. <groan> I showed up in, like, pajama pants... no slides... and just stood there smiling like a deer in headlights.. My group? Yeah—they never let me live it down.. <chuckle>",
]

def generate_and_save(prompt: str, index: int, voice: str):
    print(f"Starting generation for prompt {index}")
    start_time = time.monotonic()

    syn_tokens = model.generate_speech(prompt=prompt, voice=voice, max_tokens=8192)

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
    model = OrpheusModel(model_name="heydryft/Orpheus-3b-FT-AWQ", tokenizer="heydryft/Orpheus-3b-FT-AWQ")

    start_all = time.monotonic()
    results = []
    print("Starting generation for prompts")
    voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_and_save, prompt, idx, voices[idx % len(voices)]) for idx, prompt in enumerate(prompts)]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.monotonic() - start_all
    print("\nSummary:")
    for idx, elapsed, duration in sorted(results):
        print(f"Prompt {idx}: Generation Time = {elapsed:.2f}s, Audio Duration = {duration:.2f}s")
    print(f"Total time for all: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
