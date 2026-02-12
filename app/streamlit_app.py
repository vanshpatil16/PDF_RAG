import streamlit as st
import os
import tempfile
from src.ingestion.pdf_loader import load_pdf_sections
from src.ingestion.chunker import chunk_sections
from src.embeddings.embedder import BGEEmbedder
from src.retriever.section_retriever import FAISSRetriever
from src.generation.answer_generator import QwenAPIGenerator
from src.generation.prompt import get_qwen_messages, SECTION_AWARE_SYSTEM_PROMPT, format_context_chunks
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Section-Aware PDF RAG", layout="wide")

st.title("📄 Section-Aware PDF RAG")
st.markdown("Upload a PDF and ask questions grounded in specific sections of the document.")

# Initialize session state for RAG components
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generator" not in st.session_state:
    st.session_state.generator = QwenAPIGenerator()
if "embedder" not in st.session_state:
    st.session_state.embedder = BGEEmbedder()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Sidebar for file upload
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload research paper (PDF)", type="pdf")
    
    if uploaded_file and st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Processing PDF and building index..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # Ingestion pipeline
                sections = load_pdf_sections(tmp_path)
                chunks = chunk_sections(sections)
                
                # Embedding pipeline
                texts = [c["text"] for c in chunks]
                embeddings = st.session_state.embedder.encode(texts)
                
                # Vector Store pipeline
                retriever = FAISSRetriever(dimension=embeddings.shape[1])
                retriever.add_documents(chunks, embeddings)
                
                st.session_state.retriever = retriever
                st.session_state.processed_file = uploaded_file.name
                st.success(f"Indexed {len(chunks)} chunks from {len(sections)} sections!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
if st.session_state.processed_file:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieval
                query_embedding = st.session_state.embedder.encode_queries([prompt])
                relevant_chunks = st.session_state.retriever.retrieve(query_embedding, k=4)
                
                # Format context
                context = format_context_chunks(relevant_chunks)
                
                # Generation
                messages = get_qwen_messages(SECTION_AWARE_SYSTEM_PROMPT, prompt, context)
                response = st.session_state.generator.generate_response(messages)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a PDF file in the sidebar to start.")
