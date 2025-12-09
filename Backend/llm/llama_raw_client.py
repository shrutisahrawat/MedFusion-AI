print("Script started")
print("Sending request to Ollama...")

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

def call_llama_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"response": "...", ...}
    return data.get("response", "").strip()


if __name__ == "__main__":
    system = (
        "You are a medical *education* assistant. "
        "You NEVER give real diagnosis or prescriptions. "
        "Always add: 'This is not medical advice.'"
    )
    user = "Explain what pneumonia is to a 2nd-year MBBS student."

    full_prompt = f"{system}\n\nUser: {user}\nAssistant:"
    print(call_llama_ollama(full_prompt))
