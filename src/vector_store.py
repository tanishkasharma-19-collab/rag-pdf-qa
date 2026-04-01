from langchain_community.vectorstores import FAISS
import os

def create_or_load_vector_db(chunks, embeddings):
    if os.path.exists("faiss_index"):
        print("Loading existing FAISS index...")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("faiss_index")
    return db

def create_vector_db(chunks, embeddings):
    db = FAISS.from_documents(chunks, embeddings)
    return db