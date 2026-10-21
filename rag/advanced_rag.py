"""
Advanced RAG (Retrieval-Augmented Generation) patterns.
MultiQuery, self-query, compression and hybrid search.
"""
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from dotenv import load_dotenv
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

pdf_path = Path(__file__).parent / "astrology.pdf"   # same folder as this script

loader = PyPDFLoader(str(pdf_path))
pages = loader.load() 

# pages = loader.load()
# print("pages:", len(pages))
# print("chars per page:", [len(p.page_content) for p in pages])


INFO_BURIED = [
    Document(
        page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK

Founded in 2018 by three Stanford graduates, ACME AI Solutions began as a
small consulting firm helping enterprises adopt machine learning. Our first
office was a converted garage in Palo Alto, and we had just two laptops and
a dream. The early days were challenging - we survived on instant ramen and
the occasional pizza from the client meetings.

In 2019, we secured our first major contract with a Fortune 500 retailer,
helping them build a recommendation engine. This led to rapid growth and we
moved to a proper office space in San Francisco. By 2020, we had grown to
50 employees and opened offices in Austin and Seattle.

Our current technology stack has evolved significantly over the years. For
backend services, we use Python and FastAPI. Our data pipeline runs on
Apache Spark and Airflow. For frontend, we've standardized on React and
TypeScript.

LangChain is a framework for building LLM applications. It provides tools
for prompts, chains, agents, and memory. LangChain supports multiple LLM
providers including OpenAI, Anthropic, and local models like Llama.

The company culture at ACME emphasizes work-life balance. We offer unlimited
PTO, which most employees use for an average of 25 days per year. Our
engineering teams follow agile methodology with two-week sprints.

Our revenue has grown consistently, from $2M in 2019 to $45M in 2023. We
project $70M for 2024, driven by our new enterprise AI platform. The company
went through Series B funding in 2022, raising $80M at a $500M valuation.

Employee benefits include comprehensive health insurance through Aetna, a
401(k) with 4% matching, and a generous equity package.""",
        metadata={"source": "acme_company_overview.pdf"},
    ),
    Document(
        page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4

Chapter 1: System Architecture Overview

The ACME AI Platform is built on a microservices architecture deployed on
AWS EKS (Elastic Kubernetes Service). Each microservice is containerized
using Docker and orchestrated by Kubernetes. We use Istio as our service
mesh for traffic management and observability.

Our database layer consists of PostgreSQL for transactional data, Redis
for caching, and Pinecone for vector storage. All databases are deployed
in high-availability configurations with automatic failover.

Chapter 2: Authentication and Authorization

User authentication is handled through Auth0, supporting both SSO via SAML
2.0 and OAuth 2.0 flows. We implement role-based access control (RBAC) with
four default roles: Admin, Developer, Analyst, and Viewer.

Chapter 3: AI Framework Integration

LangGraph is a library for building stateful, multi-actor applications with
LLMs. Key features include state management, cycles and loops, human-in-the-
loop workflows, and persistence. LangGraph extends LangChain for complex
agent architectures.

Chapter 4: Monitoring and Logging

We use DataDog for application performance monitoring (APM) and log
aggregation. All services emit structured JSON logs that are collected and
indexed for searching. Alert thresholds are configured for latency (p99 >
500ms), error rates (> 1%), and resource utilization (CPU > 80%).

Chapter 5: Disaster Recovery

Our disaster recovery plan includes daily database backups stored in S3
with cross-region replication. RTO is 4 hours, and RPO is 1 hour.""",
        metadata={"source": "technical_docs_v2.4.pdf"},
    ),
]


# Sample knowledge base for demos
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

# long_doc = Document(
#         page_content="""
# # Complete Guide to Building AI Agents

# ## Chapter 1: Introduction to AI Agents

# AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

# The key components of an AI agent include:
# - A language model for reasoning
# - Tools for interacting with external systems
# - Memory for maintaining context
# - A planning mechanism for complex tasks

# ## Chapter 2: Agent Frameworks

# Several frameworks exist for building AI agents:

# LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

# LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

# CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

# ## Chapter 3: Production Considerations

# Deploying agents to production requires careful attention to:
# - Error handling and fallbacks
# - Token usage optimization
# - Observability and tracing
# - Security and access control
# - State persistence and recovery

# LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
#         """,
#         metadata={"source": "ai_agents_guide.md"},
#     )

long_doc = Document(
    page_content="\n\n".join(p.page_content for p in pages),
    metadata={"source": str(pdf_path)},
)
# print(repr(long_doc.page_content[:500]))

def create_base_vectorstore():
    """Create a base vector store for demos."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="demo_vectorstore",
        embedding_function=embeddings,
    )
    vector_store.add_documents(INFO_BURIED)
    return vector_store

def create_multi_query_retriever():
    """Create a MultiQueryRetriever for demos."""
    print("=" * 60)
    print("Multi-Query Retriever Demo")
    print("Generate multiple queries from a single user query to improve retrieval.")
    print("=" * 60)
    vector_store = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        llm=llm,
    )

    query = "What is the history of ACME AI Solutions and its technology stack?"
    print(f"\nQuery: {query}")

    print("\nThe retriever will generate multiple query variations...")
    print("(Check INFO logs above for generated queries)\n")

    # Retrieve documents
    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        print(
            f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )

def demo_contextual_compression_retriever():
    """Demonstrate ContextualCompressionRetriever with a knowledge base."""
    vector_store = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create compressor using LLMChainExtractor
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap retriever with ContextualCompressionRetriever
    compressor_retriever = ContextualCompressionRetriever(
        base_retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        base_compressor=compressor,
    )

    query = "What frameworks can I use to build AI agents, and what are their key features?"
    print(f"\nQuery: {query}")

    # Without compression
    results_no_compression = vector_store.similarity_search_with_relevance_scores(query, k=4)
    print(f"\n--- Results without compression ---")
    for i, (doc, score) in enumerate(results_no_compression):
        print(f"\nResult {i+1} (Length: {len(doc.page_content)} chars, Relevance Score: {score}):")
        print(doc.page_content)

    # With compression
    results_with_compression = compressor_retriever.invoke(query)
    print(f"\n--- Results with compression ---")
    for i, doc in enumerate(results_with_compression):
        print(f"\nResult {i+1} (Length: {len(doc.page_content)} chars):")
        print(doc.page_content)

def demo_parent_document_retriever():
    """Demonstrate ParentDocumentRetriever with a knowledge base."""
    # Create an in-memory store and add documents
    store = InMemoryStore()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", "\t"]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", "\t"]
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name="parent_doc_demo",
        embedding_function=embeddings,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    retriever.add_documents([long_doc])

    query = "What are the lucky numbers for Capricorn Makaram?"

    print(f"\nQuery: {query}")

    # Regular retrieval (would get small chunks)
    child_docs = vector_store.similarity_search(query, k=1)
    print(f"\n--- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content}...")


if __name__ == "__main__":
    # demo_parent_document_retriever()
    # demo_contextual_compression_retriever()
    create_multi_query_retriever()