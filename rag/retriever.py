from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class vectorRetriver:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.encoder=SentenceTransformer(embedding_model_name)
        self.index=None
        self.documents=[]
    
    def build_index(self,documents):
        """
        Builds a FAISS index from the provided documents.

        Args:
            documents (List[str]): List of document strings to index.
        """
        embeddings=self.encoder.encode(documents,convert_to_numpy=True)
    def retrieve(self,query,k=5):
        """
        Retrieves the top-k most similar documents to the query.

        Args:
            query (str): The query string.
            k (int): Number of top documents to retrieve.
        """
        query_embedding=self.encoder.encode([query],convert_to_numpy=True)
        distances,indices=self.index.search(query_embedding,k)
        retrieved_docs=[self.documents[idx] for idx in indices[0]]
        return retrieved_docs
        