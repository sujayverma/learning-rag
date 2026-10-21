"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile


# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = init_chat_model(
    model="gpt-4o",
    temperature=0.2,
)

def create_kb():
    """Create a Vector store from Knowledge Base"""

    # Split the knowledge base into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = Document(
        page_content=KNOWLEDGE_BASE, 
        metadata={"source": "langchain_knowledege_base.md"}
    )

    chunks = text_splitter.split_documents([docs])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=tempfile.mkdtemp()
    )

    return vector_store

def demo_basic_rag():

    vector_store = create_kb()
    retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Define a prompt template for the RAG pipeline
    prompt_template = ChatPromptTemplate.from_template(
        """
    Answer the Question based on the following context:

    {context}
    
    Question: {question}

    Answer:

    Make sure to answer in concise and clear manner.
    and if you don't know the answer, just say "I don't know" and don't try to make up an answer."""
    )

    # format retrived documents into a single string
    def format_documents(documents: List[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in documents])
    

    # RAG Chain
    rag_chain = (
        {"context": retriver | format_documents, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Test the RAG Chain with a sample question
    questions = ["What is LangChain?",
                 "Who created LangChain?",
                 "What is LangGraph used for?",
                 "Who am I?"
                ]
    
    print("=== RAG Pipeline Demo ===")
    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

def demo_basic_rag_with_sources():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context below. Include which sources you used.

        Context:
        {context}

        Question: {question}

        Answer (include sources):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")

if __name__ == "__main__":
    demo_basic_rag()
    demo_basic_rag_with_sources()

