import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Anthropic client
anthropic = Anthropic()

# Function to process multiple file types
@st.cache_resource
def process_files(file_paths):
    documents = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue
        
        try:
            loaded_docs = loader.load()
            if not isinstance(loaded_docs, list):
                loaded_docs = [loaded_docs]
            for doc in loaded_docs:
                if isinstance(doc, str):
                    documents.append(Document(page_content=doc, metadata={"source": file_path}))
                elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    documents.append(doc)
                else:
                    st.warning(f"Skipping invalid document from {file_path}")
        except Exception as e:
            st.error(f"Error loading file {file_path}: {str(e)}")
            continue
    
    if not documents:
        raise ValueError("No valid documents were loaded.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Filter complex metadata
    for doc in texts:
        doc.metadata = filter_complex_metadata(doc.metadata)
    
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    
    return db

# Function to create chatbot
def create_chatbot(db):
    def get_relevant_documents(query):
        return db.similarity_search(query)
    
    def generate_answer(query, relevant_docs, chat_history):
        context = "\n".join([doc.page_content for doc in relevant_docs])
        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            {"role": "assistant", "content": "I understand. I'll use the provided context to answer the question."}
        ]
        
        for past_query, past_answer in chat_history:
            messages.append({"role": "user", "content": past_query})
            messages.append({"role": "assistant", "content": past_answer})
        
        if messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": query})
        
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            system="You are a helpful assistant. Use the provided context to answer the user's question.",
            messages=messages
        )
        return response.content[0].text
    
    return get_relevant_documents, generate_answer

# Streamlit app
def main():
    st.set_page_config(page_title="Mukggu", page_icon="🤖")
    st.title("Mukggu bot")

    # File upload section
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or XLSX files", type=['pdf', 'docx', 'xlsx', 'xls'], accept_multiple_files=True)
    
    if uploaded_files:
        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        # Process files and create chatbot
        with st.spinner("Processing uploaded files..."):
            try:
                db = process_files(file_paths)
                get_relevant_documents, generate_answer = create_chatbot(db)
                st.success("Files processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing files: {str(e)}")
                return

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Q"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("먹꾸"):
                st.markdown(prompt)

            # Get chatbot response
            with st.spinner("노예가 생각하고 있습니다..."):
                relevant_docs = get_relevant_documents(prompt)
                response = generate_answer(prompt, relevant_docs, [(msg["content"], st.session_state.messages[i+1]["content"]) 
                                                                   for i, msg in enumerate(st.session_state.messages[:-1:2])])

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("노예"):
                st.markdown(response)

        # Clean up temporary files
        for file_path in file_paths:
            os.remove(file_path)
        os.rmdir(temp_dir)
    else:
        st.info("Please upload some files to start the chatbot.")

if __name__ == "__main__":
    main()
