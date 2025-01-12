import cohere
import streamlit as st
from typing import List, Dict

class CohereEmbedder:
    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        self.co = cohere.Client(api_key)
        self.model = model
    
    def embed_texts(self, texts: List[str]) -> Dict:
        """
        Embed a list of texts using Cohere's API
        """
        try:
            progress_text = "מבצע Embedding..."
            progress_bar = st.progress(0, text=progress_text)
            
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.co.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document",
                    embedding_types=["float"]
                )
                all_embeddings.extend(response.embeddings.float_)
                
                progress = min((i + batch_size) / len(texts), 1.0)
                progress_bar.progress(progress, text=f"{progress_text} {int(progress * 100)}%")
            
            progress_bar.empty()
            return {
                "embeddings": all_embeddings,
                "texts": texts
            }
        except Exception as e:
            st.error(f"Error during embedding: {str(e)}")
            return None

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text
        """
        try:
            response = self.co.embed(
                texts=[query],
                model=self.model,
                input_type="search_query",
                embedding_types=["float"]
            )
            return response.embeddings.float_[0]
        except Exception as e:
            st.error(f"Error during query embedding: {str(e)}")
            return None 