# agent.py (Final Version with AgentExecutor)

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# New imports for the standard agent
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. INITIALIZE COMPONENTS ONCE ---
print("Initializing RAG components...")
load_dotenv()

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory='db', embedding_function=embeddings_model)
reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
base_retriever = db.as_retriever(search_kwargs={"k": 10})
compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=compression_retriever,
)
print("Initialization complete.")

# --- 2. DEFINE YOUR TOOLS ---

@tool
def document_search_tool(query: str):
    """Searches the internal document database to answer a question about US occupations, Frey and Osborne."""
    print(f"\n--- Calling Document Search Tool with query: '{query}' ---")
    result = qa_chain.invoke({"query": query})
    return result["result"]

web_search_tool = TavilySearchResults(max_results=3)
tools = [document_search_tool, web_search_tool]

# --- 3. CREATE THE AGENT ---

# Define the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the prompt template with a clear system message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You must use the provided tools to answer the user's questions. Once you have the answer from a tool, provide that answer directly to the user."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent itself
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the Agent Executor, which runs the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. RUN THE AGENT ---

print("\n--- Running Agent on Document-Specific Question ---")
response = agent_executor.invoke({"input": "How many US occupations did Frey and Osborne claim are at risk?"})
print("\nFinal Answer:", response["output"])

print("\n\n--- Running Agent on General Knowledge Question ---")
response = agent_executor.invoke({"input": "What is the current weather in Karnal, Haryana?"})
print("\nFinal Answer:", response["output"])