import os
import requests
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()

class HFEmbeddings(Embeddings):
    """Custom HuggingFace Inference API embeddings that work reliably with FAISS."""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        result = response.json()
        # result is a list of embeddings
        if isinstance(result, list) and len(result) > 0:
            return result
        raise ValueError(f"Unexpected embedding response: {result}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process in batches of 32
        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            embeddings = self._embed(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        result = self._embed([text])
        return result[0]

def get_embeddings():
    return HFEmbeddings()