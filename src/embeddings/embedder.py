import ollama
from loguru import logger
from src.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


class OllamaEmbedder:
    def __init__(self, model: str = OLLAMA_EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"Embedding {len(texts)} chunks with {self.model}")
        vectors = []
        for i, text in enumerate(texts):
            vec = self.embed(text)
            vectors.append(vec)
            if (i + 1) % 10 == 0:
                logger.debug(f"  {i+1}/{len(texts)} done")
        return vectors

    def health_check(self) -> bool:
        try:
            self.embed("health check")
            return True
        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")
            return False
