import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# --- 1. Load Documents ---

data_dir = "docs"
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(data_dir, filename))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages from PDF documents.")

# ---2. chunk docs ----

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(documents)
print(f"Split documents into {len(chunked_documents)} chunks.")

# --- 3. Create Embeddings & Store in ChromaDB ---
# Initialize the HuggingFaceEmbeddings model

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
persist_directory = 'db'

# Create the Chroma vector store with the documents and embeddings
# This will create and save the DB to the 'db' directory

db = Chroma.from_documents(
    documents = chunked_documents,
    embedding = embeddings_model,
    persist_directory=persist_directory
)

print("Successfully created and saved the vector store.")