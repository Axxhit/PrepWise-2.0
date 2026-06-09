import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

#--base class -----------
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, prompt: str, temperature: float = 0.7) -> dict:
        """
        Returns:
        {
        "text" : str,
        "model" : str,
        "input_tokens" : int,
        "output_tokens" : int,
        }
        """
        pass

#gemini client------------

class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-3.5-flash"):
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def chat(self, prompt: str, temperature: float = 0.7) -> dict:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=500
            )
        )
        usage = response.usage_metadata
        
        return {
            "text": response.text.strip(),
            "model": self.model,
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
        }
    
#groq client----------------
class GroqClient(BaseLLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def chat(self, prompt: str, temperature: float = 0.7) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content.strip(),
            "model": self.model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
    
    # ── groq/mixtral client ───────────────────────────────────────
class MixtralClient(BaseLLMClient):
    def __init__(self):
        self.model = "mixtral-8x7b-32768"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def chat(self, prompt: str, temperature: float = 0.7) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content.strip(),
            "model": self.model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }

# ── factory — swap via env var ────────────────────────────────
def get_client(provider: str = None) -> BaseLLMClient:
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        return GeminiClient()
    elif provider == "llama":
        return GroqClient()
    elif provider == "mixtral":
        return MixtralClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
    

# ── quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "Generate one technical interview question about Random Forest classifiers."

    print(f"\n{'='*50}")
    print(f"Provider: gemini")
    client = get_client("gemini")
    result = client.chat(prompt)
    print(f"Model    : {result['model']}")
    print(f"Tokens   : {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"Output   : {result['text'][:150]}...")