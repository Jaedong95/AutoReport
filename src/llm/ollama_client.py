import ollama
from loguru import logger
from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL


class OllamaClient:
    def __init__(self, model: str = OLLAMA_LLM_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    def generate_stream(self, prompt: str, system: str | None = None,
                        temperature: float = 0.3):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    def list_models(self) -> list[str]:
        resp = self.client.list()
        return [m["name"] for m in resp.get("models", [])]

    def health_check(self) -> bool:
        try:
            models = self.list_models()
            logger.info(f"Ollama available models: {models}")
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
