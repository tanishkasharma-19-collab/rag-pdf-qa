def retrieve_docs(db, query):
    docs = db.similarity_search(query, k=3)

    print("\n--- Retrieved Context ---")
    for i, d in enumerate(docs):
        print(f"\nChunk {i+1}:")
        print(d.page_content[:300])

    return docs