import cohere
import streamlit as st
from typing import List, Dict

class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self.co = cohere.Client(api_key)
        self.model = model
    
    def rerank(self, query: str, documents: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Rerank documents based on query relevance
        """
        try:
            # Extract text from documents
            docs = [doc["text"] for doc in documents]
            
            response = self.co.rerank(
                model=self.model,
                query=query,
                documents=docs,
                top_n=top_n
            )
            
            # Reorder documents based on reranking results
            reranked_docs = []
            for result in response.results:
                idx = result.index
                reranked_docs.append({
                    **documents[idx],
                    "rerank_score": result.relevance_score
                })
            
            return reranked_docs
            
        except Exception as e:
            st.error(f"Error during reranking: {str(e)}")
            return documents[:top_n]  # Return top-k without reranking in case of error 