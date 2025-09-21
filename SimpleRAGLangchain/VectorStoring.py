#need to store the chunks embedded document to vector store we r using Pinecone.
from pathlib import Path
import os, sys, pickle, hashlib
from uuid import uuid4
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from Exception import CustomException
from loggers import logger

load_dotenv()

# ************************** PineCone Setup ************************************

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-book-index"
hash_file = "file_hashes.pkl"


# ===== Helper Functions =====
def get_file_hash(file_path):
    """Generate MD5 hash for a file"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_hashes():
    """Load saved file hashes"""
    if os.path.exists(hash_file):
        with open(hash_file, "rb") as f:
            return pickle.load(f)
    return {}


def save_hashes(hashes):
    """Save updated file hashes"""
    with open(hash_file, "wb") as f:
        pickle.dump(hashes, f)


# ===== Main Function =====
def upload_to_pinecone(file_path, chunks):
    """Check hash, create index if not exist, upload chunks"""
    try:
        emb_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # 🔹 Load existing hashes
        file_hashes = load_hashes()
        file_hash = get_file_hash(file_path)

        # 🔹 Skip if already uploaded
        if file_hash in file_hashes:
            logger.info(f"✅ {file_path} already uploaded. Skipping embedding upload.")
        else:
            logger.info(f"📄 Uploading new file: {file_path}")

            # 🔹 Ensure index exists
            existing_indexes = [idx["name"] for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    tags={"environment": "RAGdevelopment"}
                )
                logger.info(f"🆕 Created Pinecone index: {index_name}")
                import time
                time.sleep(10)  # wait for index to be ready
            else:
                logger.info(f"ℹ️ Index {index_name} already exists. Skipping creation.")

            # 🔹 Get index reference
            index = pc.Index(index_name)

            # 🔹 Generate UUIDs for chunks
            uuids = [str(uuid4()) for _ in range(len(chunks))]

            # 🔹 Upload chunks
            vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=emb_model,
                ids=uuids,
                index_name=index_name
            )
            logger.info(f"📤 Uploaded {len(chunks)} chunks to Pinecone")

            # 🔹 Save hash locally
            file_hashes[file_hash] = str(file_path)
            save_hashes(file_hashes)
            logger.info(f"💾 File hash saved for {file_path}")

        # 🔹 Always return vectorstore
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=emb_model)
        return vector_store

    except Exception as e:
        raise CustomException(e, sys)
