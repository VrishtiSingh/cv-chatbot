import os
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import streamlit as st




# Step 2: Load Word document (.docx)
loader = UnstructuredWordDocumentLoader("cv_vrish.docx")
documents = loader.load()

# Step 3: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Step 4: Create embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Step 5: Setup conversational retrieval chain
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Step 6: Streamlit UI setup
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
