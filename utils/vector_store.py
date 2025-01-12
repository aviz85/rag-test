import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
import streamlit as st

class VectorStore:
    def __init__(self, dimension: int = 1024, save_dir: str = "vector_store"):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []
        self.save_dir = save_dir
        
        # נסה לטעון אינדקס קיים
        self.load_if_exists()
    
    def add_texts(self, embeddings: List[List[float]], texts: List[str], metadata: List[Dict] = None):
        """Add texts and their embeddings to the vector store"""
        if metadata is None:
            metadata = [{} for _ in texts]
            
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        # שמור אוטומטית אחרי כל הוספה
        self.save()
    
    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """Search for most similar texts"""
        query_vector_np = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])
                })
        return results
    
    def save(self):
        """Save index and metadata to disk"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # שמור FAISS index
        faiss.write_index(self.index, os.path.join(self.save_dir, "index.faiss"))
        
        # שמור טקסטים ומטאדאטה
        with open(os.path.join(self.save_dir, "data.pkl"), "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata
            }, f)
            
        st.success("Vector store נשמר בהצלחה!")
    
    def load_if_exists(self):
        """Load index and metadata if they exist"""
        index_path = os.path.join(self.save_dir, "index.faiss")
        data_path = os.path.join(self.save_dir, "data.pkl")
        
        if os.path.exists(index_path) and os.path.exists(data_path):
            try:
                # טען FAISS index
                self.index = faiss.read_index(index_path)
                
                # טען טקסטים ומטאדאטה
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
                    self.texts = data["texts"]
                    self.metadata = data["metadata"]
                    
                st.success(f"נטענו {len(self.texts)} מסמכים מהדיסק!")
            except Exception as e:
                st.error(f"שגיאה בטעינת Vector store: {str(e)}") 