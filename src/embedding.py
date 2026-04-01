import os
import hashlib
import math
from typing import List
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

load_dotenv()

class TFIDFEmbeddings(Embeddings):
    """
    Lightweight TF-IDF style embeddings using no external ML libraries.
    Works on any Python version with zero dependencies.
    """
    
    def __init__(self, dim=384):
        self.dim = dim
    
    def _text_to_vector(self, text: str) -> List[float]:
        # Normalize text
        text = text.lower().strip()
        words = text.split()
        
        # Create a sparse vector using word hashing
        vector = [0.0] * self.dim
        word_counts = {}
        
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Hash each word to a vector position
        for word, count in word_counts.items():
            # Primary hash
            h1 = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            # Secondary hash for bigrams
            tf = count / max(len(words), 1)
            idf = math.log(1 + 1 / (1 + count))
            weight = tf * idf
            
            vector[h1] += weight
            
            # Also add character ngrams for better matching
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                h2 = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % self.dim
                vector[h2] += weight * 0.3
        
        # L2 normalize
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)

def get_embeddings():
    return TFIDFEmbeddings(dim=384)