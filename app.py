import os
import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

# Load Word document (.docx)
loader = Docx2txtLoader("cv_vrish.docx")
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Setup conversational retrieval chain with GPT-3.5 Turbo
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Streamlit UI
st.title("CV Interview Chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

query = st.text_input("Ask a question about the CV:")

if query:
    result = qa({"question": query, "chat_history": st.session_state["chat_history"]})
    st.session_state["chat_history"].append((query, result["answer"]))
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
