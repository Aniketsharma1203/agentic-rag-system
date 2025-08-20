# agentic-rag-system

Project Title: Multi-Tool RAG Agent
Overview
This project is a sophisticated, multi-tool AI agent designed to answer complex questions by intelligently drawing from both private, static documents and the real-time, public internet. The agent acts as a central reasoning engine, capable of analyzing a user's query and autonomously selecting the appropriate tool to find the most accurate answer.

The core of the system is an advanced Retrieval-Augmented Generation (RAG) pipeline, optimized for accuracy with a re-ranking stage. This pipeline is encapsulated as a custom tool, which the agent can use alongside a standard web search tool. This architecture solves a critical problem: bridging the gap between siloed internal knowledge and the vast, dynamic information available on the web.

Key Features
Intelligent Tool Routing: The agent uses a large language model (LLM) to reason about user queries and dynamically choose the best tool for the job—either querying internal documents or searching the web.

Advanced RAG Pipeline: The internal document search is powered by a state-of-the-art RAG system. It uses local Hugging Face embeddings for privacy and efficiency, a ChromaDB vector store for fast retrieval, and a Cohere re-ranker to ensure only the most relevant context is used.

Modular and Extensible: The agentic framework, built with LangChain's AgentExecutor, is inherently modular. New tools can be easily added to expand the agent's capabilities without altering the core logic.

Verifiable and Trustworthy: All answers from the internal document tool are grounded in the provided source material, ensuring responses are accurate and verifiable. The verbose output of the agent provides a clear chain of thought for its decisions and actions.

Architecture and Workflow
The system is built on a modern, agentic architecture that progresses from data ingestion to intelligent, tool-based reasoning.

1. Data Ingestion (Indexing):
The process begins with the ingest.py script, which prepares the private knowledge base. Documents are loaded, split into manageable chunks, and then converted into numerical representations (embeddings) using a local all-MiniLM-L6-v2 model. These embeddings are stored in a persistent ChromaDB vector store.

2. The RAG Tool (Internal Knowledge):
The core of the agent's internal knowledge is a sophisticated RAG pipeline that was developed and optimized:

Retrieval: A user's query is vectorized, and a similarity search is performed against ChromaDB to fetch the top k most likely document chunks.

Re-ranking: To enhance accuracy, these k chunks are passed through a Cohere re-ranker. This model re-orders the chunks based on true relevance to the query, filtering out noise and ambiguity.

Generation: The top n re-ranked chunks are then passed as context to an LLM (e.g., GPT-4o mini), which generates a final, synthesized answer. This entire pipeline is wrapped in a single, callable document_search_tool.

3. The Agent Executor (The Brain):
The agent.py script uses LangChain's AgentExecutor as the runtime for the agent. The executor manages the reasoning loop:

It receives the user's input and, guided by a system prompt, presents it to the LLM "brain."

The LLM decides whether to respond directly or use a tool. It has access to the custom document_search_tool and a Tavily web search tool.

If a tool is chosen, the AgentExecutor calls it, gets the result, and feeds it back into the loop.

This continues until the LLM determines it has enough information to provide a final, comprehensive answer to the user.

Technology Stack
Core Frameworks: LangChain, LangGraph, AgentExecutor

LLMs & Models: OpenAI GPT-4o mini (for reasoning), Cohere (for re-ranking)

Embeddings: Hugging Face all-MiniLM-L6-v2 (local, open-source)

Vector Database: ChromaDB (local, persistent)

Tools & APIs: Tavily AI (for web search)

Language: Python

How to Run This Project
To replicate this project, follow these steps:

1. Prerequisites:

Python 3.9+

Git

2. Clone the Repository:

Bash

git clone https://github.com/your-username/multi-tool-rag-agent.git
cd multi-tool-rag-agent
3. Setup Environment:

Create and activate a virtual environment:

Bash

python -m venv venv
source venv/bin/activate
Create a requirements.txt file from your project:

Bash

pip freeze > requirements.txt
Install the dependencies:

Bash

pip install -r requirements.txt
4. Configure API Keys:

Create a .env file in the root directory.

Add your API keys to the .env file:

OPENAI_API_KEY=sk-YourOpenAIKey...
COHERE_API_KEY=YourCohereKey...
TAVILY_API_KEY=tvly-YourTavilyKey...
5. Run the Application:

Place your PDF documents in a docs/ folder.

First, run the ingestion script to build the vector database:

Bash

python ingest.py
Then, run the agent to start asking questions:

Bash

python agent.py
