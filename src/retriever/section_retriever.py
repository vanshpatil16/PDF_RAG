import faiss
import numpy as np
from typing import List, Dict

class FAISSRetriever:
    def __init__(self, dimension: int = 384):
        # Default dimension for bge-small-en-v1.5 is 384
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Adds documents and their corresponding embeddings to the index.
        documents should be a list of dicts with 'text' and 'metadata'.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must match.")
        
        # Convert embeddings to float32 as required by FAISS
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.metadata.extend(documents)

    def retrieve(self, query_embedding: np.ndarray, k: int = 4) -> List[Dict]:
        """
        Retrieves top k documents for a given query embedding.
        """
        query_embedding = query_embedding.astype('float32')
        
        # FAISS search returns distances and indices
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.metadata):
                results.append(self.metadata[i])
        
        return results

if __name__ == "__main__":
    # Quick sanity check
    retriever = FAISSRetriever(dimension=2)
    docs = [
        {"text": "doc1", "metadata": {"id": 1}},
        {"text": "doc2", "metadata": {"id": 2}}
    ]
    embs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype='float32')
    retriever.add_documents(docs, embs)
    
    query = np.array([[1.1, 1.1]], dtype='float32')
    results = retriever.retrieve(query, k=1)
    print(f"Retrieved: {results[0]['text']}")
