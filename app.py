from tts_orpheus import OrpheusModel
import wave
import time

model = OrpheusModel(model_name ="canopylabs/orpheus-tts-0.1-finetune-prod")
prompt = '''Hi how you doin'''

start_time = time.monotonic()
syn_tokens = model.generate_speech(
   prompt=prompt,
   voice="tara",
)

with wave.open("output.wav", "wb") as wf:
   wf.setnchannels(1)
   wf.setsampwidth(2)
   wf.setframerate(24000)

   total_frames = 0
   chunk_counter = 0
   for audio_chunk in syn_tokens: # output streaming
      chunk_counter += 1
      frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
      total_frames += frame_count
      wf.writeframes(audio_chunk)
   duration = total_frames / wf.getframerate()

   end_time = time.monotonic()
   print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")