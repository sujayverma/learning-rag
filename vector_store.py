from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),

]

def chroma_basics():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create vector store from documents.
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings,
            persist_directory=temp_dir
        )

        print(f"Vector Store Created {vectorstore._collection.count()} documents and persisted.")

        query = "What is LangChain?"
        results = vectorstore.similarity_search(query, k=3)

        print(f"Top 3 results for query: '{query}'")

        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

def similiarity_search_with_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a vector store from documents
        vector_store = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings,
            persist_directory=temp_dir
        )

        # Perform similarity search with score.
        query = "Explain Vector Stores."
        results_with_score = vector_store.similarity_search_with_score(query, k=3)

        print(f"Top 3 results for query: '{query}'")

        for i, (doc, score) in enumerate(results_with_score):
            final_score = 1 / (1 + score)
            print(f"Result {i+1}: {doc.page_content} (Score: {final_score: .4f}, Source: {doc.metadata['source']})")

def metadata_filtering_example():
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings,
            persist_directory=temp_dir
        )

        query = "What are vector stores?"
        metadata_filter = {"topic": "database"}
        results = vector_store.similarity_search(query, k=3, filter=metadata_filter)

        print(f"Top 3 results for query: '{query}' with metadata filter: {metadata_filter}")

        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")



if __name__ == "__main__":
    # chroma_basics()    
    # similiarity_search_with_metadata()
    metadata_filtering_example()