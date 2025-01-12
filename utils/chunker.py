import tiktoken
from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text"""
        return len(self.encoding.encode(text))
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks, each with metadata
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            # Get chunk tokens
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            
            # Decode chunk
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "start_idx": i,
                    "end_idx": chunk_end,
                    **(metadata or {})
                }
            }
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            i += self.chunk_size - self.chunk_overlap
            if i >= len(tokens):
                break
        
        return chunks 