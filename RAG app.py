import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextsplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPrompTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

#load enviroment variables
load_dotenv()

#set up Groq API key
#groq_api_key = os. getenv("GROQ_API_KEY")

groq_api_key = st.getenv["GROQ_API_KEY"]

st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.image("image.png")
st.title("Dynamic RAG with Groq,FAISS, and Llama3")

#intialize session state for vector store and chat history
if "vector" not in st.session_state:
  st.session_state.vector = None
if "chat_history" not in st.session_state:
  st.session_state.chat.history = []

#sidebar for document upoad
with st.sidebar:
  st.header("Upload Documents")
  uploaded_files = st.file_uploader("Upoad your PDF documents", type="pdf", accept_multiple_files=True)
  if st.button("Process Documents"):
    if uplaoded_files:
      with st.spinner("processing Documents..."):
        docs= []
        for file in uploaded_files:
          #to read the file, we first write it to a temporary file
          with open(file.name,"wb") as f:
            f.write(file.getbuffer())
          loader = PyPDFLoader(file.name)
          docs.extend(loader.load())

      text_splitter= RecursiveCharacterTextSplitter(Chunk_size=1000, chunk_overlap=200)
      final_documents = text_splitter.split_documents(docs)

#us ea pre-trained model from Hugging Face for embeddings
      embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6_v2")
      st.session_state.vector = FAISS.from_documents(final_documents, embeddings)
      st.success("Documents processed successfully")
    else:
      st.warning("please upload at least one document")
      
#main chat Interface
st.header("Chat with Your Documents")

#intialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

#Create the prompt template
prompt = ChatPromptTemplate.from_template(
  """
  Answer the questions based on the provided context ony.
  Please provide the most accurate based on the question
  <context>
  {context}
  <context>
  Questions:{input}
  """
)

#Dispay previoys chat message 
for message in st.session_state.chat_history:
  with st.chat_message(message["role"]):
      st.markdown(message["content"])

#get user input
if prompt_input := st.chat_input("Ask a question about your documents..."):
  if st.session_state.vector is not none:
    with st.chat_message("user"):
      st.markdown(prompt_input)


    st.session_state.chat_history.append({"role":"user","content":prompt_input})

    with st.spinner("Thinking..."):
      document_chain = create_stuff_documents_chain(llm,prompt)
      retriver = st.session_state.vector.as_retriver()
      retriver_chain = create_retrival_chain(retriver, document_chain)

      start = time.process.time()
      response = retrival_chain.involve({"input":prompt_input})
      response_time = time.process_time() - start

      with st.chat_message("assistant"):
        st.markdown(response['answer'])
        st.info(f"Response time: {Response_time:.2f} seconds")

      st.session_state.chat_history.append({"role":"assisatnt", "content":"response['answer']})
                                            
else:
  st.warning("Please process your document before asking questions,")      
