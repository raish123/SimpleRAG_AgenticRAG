
#now splitting the document into chunks need splitter class
from langchain.text_splitter import RecursiveCharacterTextSplitter #this split the text document through herirachy way.

#need to store the chunks embedded document to vector store we r using Pinecone.
from pinecone import Pinecone

from pathlib import Path
import os,sys

from typing import List
#load the env files
from dotenv import load_dotenv

from Exception import CustomException
from loggers import logger
from DataIngestion import get_pdf_loader

load_dotenv()


# *******************splitting document into chunks ***************************
def get_chunks_document(doc:List[str]) ->List[str]:
    """
    here we are using document splitting technique
    in which we gonna used RecursiveCharacterTextSplitter this algorithm will split the document based
    on Text Structure following Heirarchy
    """
    try:
        #creating an object of RecursiveCharacterTextSplitter class.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap = 50
        )
        
        #now splitting the document into chunks.
        chunks = splitter.split_documents(documents=doc)
        
        logger.info(f"Splitting done: {len(chunks)} chunks created")
        
        return chunks
    
    except Exception as e:
        raise CustomException(e,sys)
    






