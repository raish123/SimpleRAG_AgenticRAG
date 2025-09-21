import streamlit as st
import time
from loggers import logger
from Exception import CustomException
import tempfile
from pathlib import Path

from GenerationChain import build_rag_chain
from DataIngestion import get_pdf_loader
from DocSplitting import get_chunks_document
from VectorStoring import upload_to_pinecone
from retrievers import create_retriever
import os,sys

def main():
    st.set_page_config(
        page_title="Retrieval Augmented Generation System",
        page_icon="🧊",
        layout="wide"
    )
    
    st.subheader("Retrieval Augmented Generation System")

    # Sidebar: file upload & processing
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload PDF", type=[".pdf"], accept_multiple_files=False)

        if "retriever" not in st.session_state:
            st.session_state.retriever = None
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
            
    

        if st.button("Submit") and uploaded_file:
            with st.spinner("Processing..."):
                try:
                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        file_path = Path(tmp_file.name)

                    # Load PDF & split chunks
                    docs = get_pdf_loader(file_path)
                    chunks = get_chunks_document(docs)

                    # Upload to vector store
                    vector_store = upload_to_pinecone(file_path, chunks)

                    # Create retriever & RAG chain
                    retriever = create_retriever(vector_store)
                    rag_chain = build_rag_chain(retriever)

                    # Save to session state
                    st.session_state.retriever = retriever
                    st.session_state.rag_chain = rag_chain

                    st.success("✅ Successfully Done!")

                except Exception as e:
                    st.error("❌ Error while processing document")
                    raise CustomException(e, sys)

    # Chat interface
    if st.session_state.rag_chain:
        # Display full chat history in main area
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").markdown(message)
            else:
                st.chat_message("assistant").markdown(message)

        # Chat input
        user_input = st.chat_input("Ask a question...")
        if user_input:
            # Append user input first
            st.session_state.chat_history.append(("user", user_input))
            st.chat_message("user").markdown(user_input)

            try:
                # Retrieve docs
                retrieved_docs = st.session_state.retriever.invoke(user_input)
                for i, doc in enumerate(retrieved_docs):
                    logger.info(f"Query: {user_input} | Doc {i+1}: {doc.page_content[:200]}... | Metadata: {doc.metadata}")

                # Get answer from RAG
                answer = st.session_state.rag_chain.invoke(user_input)

                # Streaming generator
                def response_generator(): 
                    for word in answer.split(): 
                        yield word + " " 
                        time.sleep(0.05)

                # Display streamed response
                with st.chat_message("assistant"): 
                    st.write_stream(response_generator())

                # Append full answer to chat history
                st.session_state.chat_history.append(("bot", answer))

            except Exception as e:
                st.error("⚠️ Something went wrong")
                raise CustomException(e, sys)



if __name__ == "__main__":
    main()
