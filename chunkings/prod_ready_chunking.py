from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()



# Sample document with distinct topics
document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""
pdf_path = Path(__file__).parent / "astrology.pdf"   # same folder as this script

loader = PyPDFLoader(str(pdf_path))
pages = loader.load() 
long_doc = Document(
    page_content="\n\n".join(p.page_content for p in pages),
    metadata={"source": str(pdf_path)},
)

def smart_chunker(
        text: str, 
        use_semantic_chunking: bool = True, 
        fallback_chunk_size: int = 400) -> list[str]:
    """Chunk text using semantic chunking with a fallback to recursive-based chunking."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if use_semantic_chunking:
        try:
            # Use semantic chunking
            chunker = SemanticChunker(embeddings=embeddings, 
                                      breakpoint_threshold_type="percentile",
                                      breakpoint_threshold_amount=90)
            chunks = chunker.split_text(text)

            max_chunk_length = 2000

            if any(len(chunk) > max_chunk_length for chunk in chunks):
                return recursive_chunker(text, chunk_size=fallback_chunk_size)

            return chunks
        except Exception as e:
            print(f"Semantic chunking failed: {e}. Falling back to recursive chunking.")
            return recursive_chunker(text, chunk_size=fallback_chunk_size)

    return recursive_chunker(text, chunk_size=fallback_chunk_size)

def recursive_chunker(text: str, chunk_size: int = 400) -> list[str]:
    """Chunk text using recursive character-based chunking."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    return splitter.split_text(text)

chunks = smart_chunker(long_doc.page_content, use_semantic_chunking=True, fallback_chunk_size=400)
print(f'Created {len(chunks)} semantic chunks')
