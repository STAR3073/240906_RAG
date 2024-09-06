from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
from transformers import Gemma2ForCausalLM
from accelerate import Accelerator
import bs4
import streamlit as st



def rag_setup(file_path, chunk_size=1000, chunk_overlap=50, k=4, weight=0.5):
    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True})

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(
        split_documents
    )
    sparse_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[weight, 1 - weight],
    )   

    # 반환값
    return ensemble_retriever




def create_rag_chain(retriever):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Context: 
    {context} 

    #Question: 
    {question}

    #Answer:"""
    )


    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = Gemma2ForCausalLM.from_pretrained(
        "rtzr/ko-gemma-2-9b-it",
        device_map={"": "cpu"},
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
