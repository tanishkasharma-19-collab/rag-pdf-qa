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
        print("FAISS index saved.")
    return db

def create_vector_db(chunks, embeddings):
    # Embed in batches to avoid API rate limits
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    
    import faiss
    import numpy as np
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain.schema import Document
    
    dim = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    vectors = np.array(all_embeddings, dtype=np.float32)
    index.add(vectors)
    
    docstore = InMemoryDocstore({str(i): Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))})
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}
    
    db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return db