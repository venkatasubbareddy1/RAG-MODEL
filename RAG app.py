# app.py

import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration and API Key Handling ---

# Set page configuration for a better layout
st.set_page_config(page_title="Chat with Your Documents", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Chat with Your Documents using Llama3 & Groq")

# Robustly load the API key from Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it.")
    st.stop()

# --- Session State Initialization ---

# Initialize session state variables if they don't exist
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# --- Sidebar for Document Upload and Processing ---

with st.sidebar:
    st.header("1. Upload Your PDFs")
    uploaded_files = st.file_uploader("You can upload multiple PDF files at once.", type="pdf", accept_multiple_files=True)

    if st.button("2. Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment. ⏳"):
                try:
                    # Create a temporary directory to store uploaded files safely
                    temp_dir = "temp_pdf_files"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)

                    docs = []
                    for file in uploaded_files:
                        temp_filepath = os.path.join(temp_dir, file.name)
                        with open(temp_filepath, "wb") as f:
                            f.write(file.getbuffer())
                        
                        loader = PyPDFLoader(temp_filepath)
                        docs.extend(loader.load())

                    # 1. Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(docs)

                    # 2. Create vector embeddings and store in FAISS
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vector = FAISS.from_documents(final_documents, embeddings)
                    
                    st.session_state.documents_processed = True
                    st.success("✅ Documents processed successfully!")
                    st.info("You can now ask questions about your documents in the main chat area.")
                
                except Exception as e:
                    st.error(f"An error occurred during document processing: {e}")
        else:
            st.warning("⚠️ Please upload at least one PDF file to process.")

# --- Main Chat Interface ---

st.header("3. Ask Questions About Your Documents")

# Initialize the language model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    Think step-by-step and provide a detailed and accurate response.
    If the answer is not available in the context, state that you cannot answer.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# Display previous chat messages from history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt_input := st.chat_input("Ask your question here..."):
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process your documents before asking questions.")
    else:
        # Add user message to chat history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Create the retrieval chain and invoke it
        with st.spinner("Thinking... 🤖"):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt_input})
            response_time = time.process_time() - start
            
            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                st.info(f"Response time: {response_time:.2f} seconds")

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
