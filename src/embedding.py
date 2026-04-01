from langchain_community.embeddings import FastEmbedEmbeddings

def get_embeddings():
    """
    Uses FastEmbed - lightweight, no torch needed, runs locally and fast.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")