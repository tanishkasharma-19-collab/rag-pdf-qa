import os
import time
import requests
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()

class HFEmbeddings(Embeddings):
    """Custom HuggingFace Inference API embeddings."""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(5):  # retry up to 5 times
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            result = response.json()
            
            # Model is loading, wait and retry
            if isinstance(result, dict) and "error" in result:
                wait = result.get("estimated_time", 20)
                print(f"Model loading, waiting {wait}s... (attempt {attempt+1})")
                time.sleep(min(wait, 30))
                continue
            
            # Valid response - list of embeddings
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                return result
            
            # Single embedding returned for single input
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], float):
                return [result]
                
            print(f"Unexpected response: {str(result)[:200]}, retrying...")
            time.sleep(5)
        
        raise ValueError("HuggingFace API failed after 5 attempts")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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