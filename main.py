from src.data_loader import load_pdf
from src.chunker import chunk_data
from src.embedding import get_embeddings
from src.vector_store import create_or_load_vector_db
from src.retriever import retrieve_docs
from src.ocr_loader import load_pdf_with_ocr
from src.groq_llm import generate_answer_groq

import os
import time

# -------- MULTI PDF LOADER --------
all_docs = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        path = os.path.join("data", file)

        try:
            docs = load_pdf(path)

            # If extracted text is very short → scanned PDF, use OCR
            if not docs or len(docs[0].page_content.strip()) < 20:
                print(f"{file} seems scanned → Using OCR")
                docs = load_pdf_with_ocr(path)

        except Exception as e:
            print(f"Normal loading failed for {file} → Using OCR. Error: {e}")
            docs = load_pdf_with_ocr(path)

        all_docs.extend(docs)

# -------- CHUNKING --------
chunks = chunk_data(all_docs)

# -------- EMBEDDINGS --------
embeddings = get_embeddings()

# -------- VECTOR DB --------
db = create_or_load_vector_db(chunks, embeddings)

print("\nSystem Ready. Ask Questions.\n")

# -------- QUERY LOOP --------
while True:
    query = input("\nQuestion: ")

    if query.lower() == "exit":
        break

    start = time.time()

    retrieved_docs = retrieve_docs(db, query)
    answer = generate_answer_groq(query, retrieved_docs)

    end = time.time()

    print("\nAnswer:", answer)
    print("Response Time:", round(end - start, 2), "seconds")