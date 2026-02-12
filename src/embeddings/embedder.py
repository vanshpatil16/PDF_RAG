import os
import numpy as np
from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class BGEEmbedder:
    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5"):
        self.client = InferenceClient(
            model=model_id,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        )

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encodes a list of queries using the Hugging Face Inference API.
        """
        # Note: Inference API's feature_extraction returns embeddings
        # For BGE, we usually want to prepend instructions, but the Inference API
        # might handle it or we can do it manually if needed.
        # Here we follow the user's requested API structure.
        embeddings = self.client.feature_extraction(queries)
        return np.array(embeddings)

    def encode(self, passages: List[str]) -> np.ndarray:
        """
        Encodes a list of passages using the Hugging Face Inference API.
        """
        embeddings = self.client.feature_extraction(passages)
        return np.array(embeddings)

if __name__ == "__main__":
    embedder = BGEEmbedder()
    test_queries = ["What is RAG?"]
    test_passages = ["Retrieval-augmented generation is a technique..."]
    
    q_emb = embedder.encode_queries(test_queries)
    p_emb = embedder.encode(test_passages)
    
    print(f"Query embedding shape: {q_emb.shape}")
    print(f"Passage embedding shape: {p_emb.shape}")
    similarity = q_emb @ p_emb.T
    print(f"Similarity: {similarity[0][0]}")
