from pathlib import Path
import os,sys

from typing import List

#load the env files
from dotenv import load_dotenv

from Exception import CustomException
from loggers import logger

load_dotenv()

from langchain_pinecone import PineconeVectorStore
from langchain_openai import  OpenAIEmbeddings

def create_retriever(vector_store: PineconeVectorStore):
    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.25}
        )
        logger.info("Retriever created successfully")
        return retriever
    except Exception as e:
        raise CustomException(e, sys)

# ********************Helper Function************
def combine_retrieve_doc(retrieve_doc: List[str]) -> str:
    """
    Combine retrieved documents into a single string
    """
    try:
        lst_doc = []
        for content in  retrieve_doc:
            lst_doc.append(content.page_content)
        
        join_doc = "\n\n".join(lst_doc)

        logger.info("Documents combined successfully")
        return join_doc
    except Exception as e:
        raise CustomException(e, sys)
