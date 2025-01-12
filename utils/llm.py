import anthropic
import streamlit as st
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.generation_config = {
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
        }
    
    def update_config(self, **kwargs):
        """Update generation config parameters"""
        self.generation_config.update(kwargs)
    
    def format_context(self, documents: List[Dict], query: str) -> str:
        """Format context and query for the LLM"""
        context = "\n\n".join([
            f"מסמך {i+1} (רלוונטיות: {doc.get('rerank_score', doc.get('score', 0)):.3f}):\n{doc['text']}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""מסמכים רלוונטיים:
{context}

שאלה: {query}

תשובה (בהתבסס אך ורק על המסמכים לעיל):"""
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _send_message(self, prompt: str, system_prompt: str) -> str:
        """Send message with retry logic"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.generation_config["max_tokens"],
                temperature=self.generation_config["temperature"],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        except Exception as e:
            st.error(f"שגיאה בשליחת ההודעה: {str(e)}")
            return "מצטער, אירעה שגיאה בעת יצירת התשובה. אנא נסה שוב."
    
    def generate_response(self, query: str, documents: List[Dict], system_prompt: str) -> str:
        """Generate response based on query and relevant documents"""
        try:
            prompt = self.format_context(documents, query)
            
            with st.spinner('מייצר תשובה...'):
                return self._send_message(prompt, system_prompt)
            
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            return "מצטער, אירעה שגיאה בעת יצירת התשובה. אנא נסה שוב." 