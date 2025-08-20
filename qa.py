import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

load_dotenv()
persist_directory = 'db'
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings_model
)

# --- 2. Create the Retriever ---

base_retriever = db.as_retriever(search_kwargs={"k": 10})

reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, 
    base_retriever=base_retriever
)

# --- 3. Create the Q&A Chain ---
# The chain combines the LLM and the retriever

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=compression_retriever, # <-- USE THE NEW RETRIEVER
    return_source_documents=True
)

query = "How many US occupations did Frey and Osborne claim are at risk of automation, and over what time frame?" # Change this to your question
result = qa_chain({"query": query})

print("Question:", query)
print("Answer:", result["result"])
print("\n--- Source Documents (Re-ranked) ---")
for doc in result["source_documents"]:
    print(f"Page {doc.metadata['page']} of {os.path.basename(doc.metadata['source'])}:")