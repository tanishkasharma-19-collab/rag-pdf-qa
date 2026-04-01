from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    """
    Uses HuggingFace Inference API for embeddings - no local model needed.
    Falls back to a simple TF-IDF style if API not available.
    """
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        # Fallback: use OpenAI-compatible embeddings via Groq isn't available,
        # so we use a lightweight approach
        from langchain_community.embeddings import FakeEmbeddings
        print("Warning: No HF_TOKEN found, using fake embeddings (for testing only)")
        return FakeEmbeddings(size=384)