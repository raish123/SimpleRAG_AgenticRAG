from GenerationChain import build_rag_chain
from DataIngestion import get_pdf_loader
from DocSplitting import get_chunks_document
from VectorStoring import upload_to_pinecone
from pathlib import Path
from retrievers import create_retriever,combine_retrieve_doc
from GenerationChain import build_rag_chain
from Exception import CustomException
from loggers import logger
import os,sys



def main():
    try:
        pdf_file = Path("Medical_book.pdf")

        # Step 1️⃣ Load PDF
        docs = get_pdf_loader(pdf_file)

        # Step 2️⃣ Split into chunks
        chunks = get_chunks_document(docs)

        # Step 3️⃣ Upload chunks to Pinecone
        vector_store = upload_to_pinecone(pdf_file, chunks)

        # Step 4️⃣ Create retriever from Pinecone
        retriever = create_retriever(vector_store)

        # Step 5️⃣ Test query
        query = "What is diabetes?"
        retrieved_docs = retriever.get_relevant_documents(query=query)

        # Step 6️⃣ Combine retrieved docs
        combined_doc = combine_retrieve_doc(retrieved_docs)
        print("\n--- Combined Retrieved Docs ---\n")
        print(combined_doc[:500], "...")  # print first 500 chars

        # Step 7️⃣ Build RAG chain
        rag_chain = build_rag_chain(retriever)

        # Step 8️⃣ Get answer from RAG chain
        answer = rag_chain.invoke(query)
        print("\n--- RAG Answer ---\n")
        print(answer)
        
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    main()