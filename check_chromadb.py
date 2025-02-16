import chromadb

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="book_recommendations")

# Fetch all stored documents
all_docs = collection.get()

print("DEBUG: Total Documents in ChromaDB:", len(all_docs["ids"]))

# Print first 5 stored entries
for i in range(min(5, len(all_docs["ids"]))):
    doc_id = all_docs["ids"][i]
    metadata = all_docs["metadatas"][i]
    embedding = all_docs["embeddings"][i]

    print(f"\nDEBUG: Document {i}")
    print(f"ID: {doc_id}")
    print(f"Metadata: {metadata}")
    print(f"Embedding Length: {len(embedding)}")

    # Check for None or bad data
    if metadata is None or not isinstance(metadata, dict):
        print(f"ERROR: Metadata for document {i} is invalid!")

    chunk_text = metadata.get("chunk", None)
    if chunk_text is None or not isinstance(chunk_text, str) or chunk_text.strip() == "":
        print(f"ERROR: Document {i} has invalid or empty 'chunk' text!")
