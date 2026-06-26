import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def call_groq(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,   
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()