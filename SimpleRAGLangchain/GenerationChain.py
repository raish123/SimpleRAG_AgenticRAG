import sys
from langchain import hub
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from Exception import CustomException
from loggers import logger
#below classes we used so user can interact with LLM Models.
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from typing import List
from retrievers import combine_retrieve_doc


def build_rag_chain(retriever):
    """
    Build the final RAG chain using retriever, prompt, and LLM
    """

    try:
        #openai model
        model2 = ChatOpenAI(
            model="gpt-3.5-turbo",temperature=0.1
        )
        
        
        # 1. Get the RAG prompt from LangChain hub
        rag_prompt = hub.pull("rlm/rag-prompt")
        logger.info("RAG Prompt pulled from hub")

      
        # 2. Output parser
        parser = StrOutputParser()

        # 3. Sequential chain (Prompt -> LLM -> Parser)
        seq_chain = rag_prompt | model2 | parser

        # 4. Parallel chain (fetch context + question passthrough)
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(combine_retrieve_doc),
            "question": RunnablePassthrough(),
        })

        # 6. Final chain (combine context + LLM)
        final_chain = parallel_chain | seq_chain

        logger.info("RAG Chain built successfully")
        return final_chain

    except Exception as e:
        raise CustomException(e, sys)
    
    
