from dotenv import load_dotenv

load_dotenv()

import os
import openai
BASE_URL = os.getenv("GROQ_API_BASE")
API_KEY = os.getenv("GROQ_API_KEY")

openai.api_key = API_KEY
openai.base_url = BASE_URL

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

def respond(message):
    messages.append({"role": "user", "content": message})
    
    response = openai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

def warmup():
    openai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}],
    )

warmup()