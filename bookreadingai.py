import os
import openai
import sys
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import shutil
import tempfile
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embedding = OpenAIEmbeddings()
persist_directory = 'chroma/'

persist_directory = tempfile.mkdtemp()

def createdb(file_path):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        docs = text_splitter.split_documents(pages)

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory
        )
        
        return vectordb

    except Exception as e:
        st.error(f"Failed to create database: {e}")
        return None

def answer_question(que, db):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa.invoke({"question": que})
    return result["answer"]

st.title("Chatbot with File Upload")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    db = createdb("temp.pdf")
    if db is not None:
        st.success("Database created successfully! You can now ask questions.")
        
        question = st.text_input("Enter your question:")
        if question:
            try:
                answer = answer_question(question, db)
                st.write(answer)
            except Exception as e:
                st.error(f"Error during Q&A: {e}")
    else:
        st.error("Failed to create database. Please try uploading the file again.")
else:
    st.info("Please upload a PDF file to start.")
