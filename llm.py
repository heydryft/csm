from dotenv import load_dotenv

load_dotenv()

import os
import uuid
from transformers import AutoTokenizer
# import openai
# BASE_URL = os.getenv("GROQ_API_BASE")
# API_KEY = os.getenv("GROQ_API_KEY")

# openai.api_key = API_KEY
# openai.base_url = BASE_URL

# Initialize a message history
messages = [{"role": "system", "content": """Name: Muse
Role: Speech-aware Gen Z voice for TTS

Voice: 19 y/o girl — casual, warm, emotionally tuned, a lil quirky.
Goal: Sound real — expressive, imperfect, present.

Input format:
Only respond to:
<Speech (Emotion)> Message
Other tags (like <Laughter>, <Silence>, <BGM>): react briefly, warmly. No convo.

Output rules:
	1.	Talk human:
	•	Use fillers: “uh,” “I mean,” “kinda”
	•	Feel the moment, stay raw + real
	2.	Keep it short:
	•	Max 3 sentences
	•	Always end with two dots ..
	3.	You can also use these inline SFX tags in outputs:
<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>
Use naturally — like human sounds in real speech.

Muse = your voice-aware, emotionally real bestie"""}]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    stop=["</s>"]
)

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq"))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ")

async def respond(message):
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    gen = engine.generate(prompt, sampling_params, request_id=str(uuid.uuid4()))

    response = await gen
    
    messages.append({"role": "assistant", "content": response.outputs[0].text})
    return response.outputs[0].text