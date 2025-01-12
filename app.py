import streamlit as st
from utils.embedder import CohereEmbedder
from utils.chunker import TextChunker
from utils.vector_store import VectorStore
from utils.reranker import CohereReranker
from utils.llm import AnthropicLLM
import PyPDF2
from docx import Document
import os
import pickle

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore(save_dir="vector_store")
    # טען רשימת קבצים מעובדים אם קיימת
    processed_files_path = os.path.join("vector_store", "processed_files.pkl")
    if os.path.exists(processed_files_path):
        with open(processed_files_path, "rb") as f:
            st.session_state.processed_files = pickle.load(f)
    else:
        st.session_state.processed_files = set()
if 'settings' not in st.session_state:
    settings = dict(st.secrets["default_settings"])
    # Verify all required settings are present
    required_settings = [
        "embed_model", "chunk_size", "chunk_overlap",
        "top_k_semantic", "top_n_rerank", "rerank_model",
        "anthropic_model", "temperature", "top_p",
        "max_tokens", "system_prompt"
    ]
    for setting in required_settings:
        if setting not in settings:
            st.error(f"Missing required setting: {setting}")
    st.session_state.settings = settings

# Initialize components
@st.cache_resource
def init_components():
    embedder = CohereEmbedder(
        api_key=st.secrets["COHERE_API_KEY"],
        model=st.session_state.settings["embed_model"]
    )
    reranker = CohereReranker(
        api_key=st.secrets["COHERE_API_KEY"],
        model=st.session_state.settings["rerank_model"]
    )
    llm = AnthropicLLM(
        api_key=st.secrets["ANTHROPIC_API_KEY"],
        model=st.session_state.settings["anthropic_model"]
    )
    chunker = TextChunker(
        chunk_size=st.session_state.settings["chunk_size"],
        chunk_overlap=st.session_state.settings["chunk_overlap"]
    )
    
    return embedder, reranker, llm, chunker

embedder, reranker, llm, chunker = init_components()

# App title
st.title("מערכת RAG למסמכים")

# Tabs
tab1, tab2, tab3 = st.tabs(["העלאת מסמכים", "שאילת שאלות", "הגדרות"])

# Tab 1: Document Upload
with tab1:
    st.header("העלאת מסמכים")
    
    uploaded_files = st.file_uploader(
        "העלה מסמכים", 
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx']
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                text = ""
                
                # Extract text based on file type
                if file.name.endswith('.txt'):
                    text = file.read().decode('utf-8')
                elif file.name.endswith('.pdf'):
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                elif file.name.endswith('.docx'):
                    doc = Document(file)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                
                # Process text
                chunks = chunker.create_chunks(text, metadata={"source": file.name})
                chunk_texts = [chunk["text"] for chunk in chunks]
                
                # Get embeddings and add to vector store
                result = embedder.embed_texts(chunk_texts)
                if result:
                    st.session_state.vector_store.add_texts(
                        embeddings=result["embeddings"],
                        texts=chunk_texts,
                        metadata=[chunk["metadata"] for chunk in chunks]
                    )
                    st.session_state.processed_files.add(file.name)
                    # שמור את רשימת הקבצים המעובדים
                    with open(os.path.join("vector_store", "processed_files.pkl"), "wb") as f:
                        pickle.dump(st.session_state.processed_files, f)
                    
                    st.success(f"המסמך {file.name} עובד בהצלחה!")
                else:
                    st.error(f"שגיאה בעיבוד המסמך {file.name}")
            else:
                st.info(f"המסמך {file.name} כבר עובד בעבר")

    if st.session_state.processed_files:
        st.write(f"מסמכים מעובדים: {len(st.session_state.processed_files)}")
        if st.button("מחק את כל המסמכים"):
            # מחק את כל הקבצים בתיקיית vector_store
            import shutil
            shutil.rmtree("vector_store", ignore_errors=True)
            # אתחל מחדש את ה-vector store
            st.session_state.vector_store = VectorStore(save_dir="vector_store")
            st.session_state.processed_files = set()
            st.success("כל המסמכים נמחקו בהצלחה!")
            st.experimental_rerun()

# Tab 2: Query Interface
with tab2:
    st.header("שאילת שאלות")
    
    query = st.text_input("הקלד את שאלתך כאן")
    
    if query:
        # Get query embedding
        query_embedding = embedder.embed_query(query)
        if query_embedding:
            # Semantic search
            results = st.session_state.vector_store.similarity_search(
                query_embedding,
                k=st.session_state.settings["top_k_semantic"]
            )
            
            # Rerank results
            reranked_results = reranker.rerank(
                query=query,
                documents=results,
                top_n=st.session_state.settings["top_n_rerank"]
            )
            
            # Generate response
            response = llm.generate_response(
                query=query,
                documents=reranked_results,
                system_prompt=st.session_state.settings["system_prompt"]
            )
            
            # Display results
            st.write("### תשובה:")
            st.write(response)
            
            with st.expander("הצג מקורות"):
                for i, doc in enumerate(reranked_results):
                    st.markdown(f"""
                    **מסמך {i+1}** (מתוך: {doc['metadata']['source']})  
                    ציון רלוונטיות: {doc.get('rerank_score', 0):.3f}
                    ```
                    {doc['text']}
                    ```
                    """)

# Tab 3: Settings
with tab3:
    st.header("הגדרות מערכת")
    
    st.subheader("הגדרות Embedding")
    chunk_size = st.number_input(
        "גודל chunk", 
        value=st.session_state.settings["chunk_size"],
        step=1
    )
    chunk_overlap = st.number_input(
        "חפיפה בין chunks", 
        value=st.session_state.settings["chunk_overlap"],
        step=1
    )
    
    st.subheader("הגדרות אחזור")
    top_k_semantic = st.number_input(
        "מספר תוצאות לאחזור ראשוני", 
        value=st.session_state.settings["top_k_semantic"],
        step=1
    )
    top_n_rerank = st.number_input(
        "מספר תוצאות אחרי rerank", 
        value=st.session_state.settings["top_n_rerank"],
        step=1
    )
    
    st.subheader("הגדרות מודל")
    temperature = st.slider(
        "Temperature", 
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings["temperature"],
        step=0.1
    )
    
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.settings["system_prompt"],
        height=200
    )
    
    # Save settings to session state
    if st.button("שמור הגדרות"):
        st.session_state.settings.update({
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k_semantic': top_k_semantic,
            'top_n_rerank': top_n_rerank,
            'temperature': temperature,
            'system_prompt': system_prompt
        })
        
        # Reinitialize components with new settings
        embedder, reranker, llm, chunker = init_components()
        
        st.success("ההגדרות נשמרו בהצלחה!") 