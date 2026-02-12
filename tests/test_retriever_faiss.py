import os
import sys
from src.ingestion.pdf_loader import load_pdf_sections
from src.ingestion.chunker import chunk_sections
from src.embeddings.embedder import BGEEmbedder
from src.retriever.section_retriever import FAISSRetriever

def test_full_retrieval_flow():
    pdf_path = "data/raw/understanding_womens_safety.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF not found at {pdf_path}")
        return

    print("1. Loading PDF sections...")
    sections = load_pdf_sections(pdf_path)
    print(f"   Loaded {len(sections)} sections.")

    print("2. Chunking sections...")
    chunks = chunk_sections(sections)
    print(f"   Created {len(chunks)} chunks.")

    print("3. Initializing BGE Embedder...")
    embedder = BGEEmbedder()
    
    print("4. Embedding chunks (this might take a moment)...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    print(f"   Embeddings shape: {embeddings.shape}")

    print("5. Indexing in FAISS...")
    retriever = FAISSRetriever(dimension=embeddings.shape[1])
    retriever.add_documents(chunks, embeddings)
    print("   Indexing complete.")

    print("\n6. Testing Retrieval...")
    query = "What does the report say about lighting and safety?"
    print(f"   Query: {query}")
    
    query_embedding = embedder.encode_queries([query])
    results = retriever.retrieve(query_embedding, k=3)
    
    print("\n   Top Results:")
    for i, res in enumerate(results):
        print(f"\n   [{i+1}] Section: {res['metadata']['section']}, Chunk: {res['metadata']['chunk_id']}")
        print(f"   Page: {res['metadata']['page']}")
        print(f"   Text Snippet: {res['text'][:150]}...")
    
    print("\nRetrieval flow verified! ✅")

if __name__ == "__main__":
    test_full_retrieval_flow()
