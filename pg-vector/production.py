import os
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# Configuration.
@dataclass
class Config:
    env_name: str = os.env.get("ENV_NAME", "local")
    if env_name == "local":
        db_url = os.getenv("LOCAL_DB_URL")
    elif env_name == "dev":
        db_url = os.getenv("SUPABASE_DB_URL")
    else:
        raise ValueError(f"Unknown ENV_NAME: {env_name}")

    collection_name: str = "ProductionCollection"

    embeddings_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.7

    # search settings
    default_k: int = 5
    min_similarity: float = 0.75


class RAGService:
    """ A simple production level service for performing retrieval-augmented generation (RAG) using PostgreSQL and pgvector."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._vectorstore = None
        self._chat_model = None
        self._chain = None

    @property
    def vectorstore(self) -> PGVector:
        if self._vectorstore is None:
            embeddings = OpenAIEmbeddings(model=self.config.embeddings_model)
            self._vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.db_url,
                use_jsonb=True,  # Use JSONB for storing vectors
            )
        return self._vectorstore

    @property
    def chain(self):
        """Lazy Initialization of the RAG chain."""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain

    def _create_chain(self):
        """ Create a simple RAG Chain"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.default_k, "min_similarity": self.config.min_similarity}
        )
        
