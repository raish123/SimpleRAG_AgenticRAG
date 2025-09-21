#Data Ingestion Fetching data from different source and splitting it.
#importing modules which is used in simple RAG Project.


import pdfplumber

#want to load document into workingspace using document loaders
from langchain_community.document_loaders import PDFPlumberLoader,TextLoader

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

load_dotenv()



# *****************************Document Loading ************************************************
def get_pdf_loader(pdf_path:Path)->list[str]:
    """
    Extracts text from all pages of a PDF file using pdfplumber and 
    stores it in a single string.
    
    """
    #creating an object of PDFPlumberLoader class
    try:
        
        loader = PDFPlumberLoader(
            file_path=pdf_path,
            extract_images=False
        )
        
        logger.info("PDFPlumberLoader Loader object Created")
        
        #now loading the pdfs in my work space.
        doc_load = loader.load() 
        logger.info(f"Document Loaded {doc_load[:2]}")
        
        
        return doc_load

        
        #now loading the pdf file into work space.
    except Exception as e:
        raise CustomException(e,sys)
    
    

