from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    hf_token = os.getenv("HF_TOKEN")
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )