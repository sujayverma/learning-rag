from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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

def detect_topics(chunk):
    """Detect which section headers appear in a chunk."""
    topics = []
    for topic in [
        "OAuth2 Authentication",
        "Rate Limiting",
        "Error Handling",
        "Webhooks",
    ]:
        if topic.lower() in chunk.lower():
            topics.append(topic)
    return topics

def print_chunks(chunks, method_name):
    """Print chunks with topic detection and clear formatting."""
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        topics = detect_topics(chunk)
        topic_label = f" | Topics: {', '.join(topics)}" if topics else ""
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars){topic_label} ---")
        # Show first 3 lines for a quick preview
        lines = chunk.strip().split("\n")
        preview = "\n".join(lines[:3])
        if len(lines) > 3:
            preview += f"\n  ... ({len(lines) - 3} more lines)"
        print(preview)


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", "\t", ". "]
)

recursive_chunks = recursive_splitter.split_text(document)

print_chunks(recursive_chunks, "RecursiveCharacterTextSplitter")

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

semantic_chunks = semantic_chunker.split_text(document)

print_chunks(semantic_chunks, "SemanticChunker")

recursive_vector_store = Chroma.from_texts(recursive_chunks, embeddings, collection_name="recursive_chunks")
semantic_vector_store = Chroma.from_texts(semantic_chunks, embeddings, collection_name="semantic_chunks")


test_query = [
    "How do I authenticate using OAuth2?",
    "What are the rate limits for the free tier?",
    "How should I handle errors returned by the API?",
    "How can I set up webhooks and verify their signatures?"
]


def test_retrival(query, vector_store, name):
    """Test retrieval from a vector store for a given query."""
    results = vector_store.similarity_search_with_score(query, k=2)
    print(f"\n{'='*60}")
    print(f"  Retrieval Test: {name}")
    print(f"  Query: {query}")
    print(f"{'='*60}")
    for i, (result, score) in enumerate(results):
        final_score = 1 / (1 + score)
        print(f"\n--- Result {i+1} ({len(result.page_content)} chars) (Score: {final_score: .4f}) ---")
        print(result.page_content)


for query in test_query:
    test_retrival(query, recursive_vector_store, "RecursiveCharacterTextSplitter")
    test_retrival(query, semantic_vector_store, "SemanticChunker")