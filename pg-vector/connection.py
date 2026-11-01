"""
Learning to connect to a PostgreSQL database using psycopg2 and pgvector in supabase. This script demonstrates how to establish a connection, create a table with vector support, and insert data into the table.
Make sure to install the required libraries before running this script:
```bash
pip install psycopg2-binary pgvector
```
"""
import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

env_name = os.getenv("ENV_NAME", "local")
print(f"Connecting to {env_name} database...")
if env_name == "local":
    db_url = os.getenv("LOCAL_DB_URL")
elif env_name == "dev":
    db_url = os.getenv("SUPABASE_DB_URL")

print(f"Database URL: {db_url}")

def connect_to_postgres(db_url):
    """Connect to PostgreSQL database using psycopg2 and pgvector."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        pg_vector = PGVector(
            embeddings=embeddings,
            collection_name="my_vector_collection",
            connection=db_url,
            use_jsonb=True,  # Use JSONB for storing vectors
        )
        print("Connection to PostgreSQL database established successfully.")
        return pg_vector
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def verify_connection(vector_store):
    """Verify the connection by checking if the collection exists."""
    from langchain_core.documents import Document
    try:
        
        test_doc = Document(
            page_content="This is a test document to verify Supabase connection.",
            metadata={"test": True},
        )
        ids = vector_store.add_documents([test_doc])
        # print(f"Test document added to the collection. Total documents: {vector_store.count_documents()}")
        print(f"Document IDs: {ids[0]}")
        verify_collection = vector_store.similarity_search("test document", k=1)
        if verify_collection:
            print("Verification successful: Retrieved document from the collection.")
            print(f"Retrieved document content: {verify_collection[0].page_content}")
        else:
            print("Verification failed: No documents retrieved from the collection.")

        # vector_store.delete(ids)
        # print("Test document deleted from the collection.")
        return True
    except Exception as e:
        print(f"Error verifying collection: {e}")
        return False

def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    vector_store = connect_to_postgres(db_url)
    if vector_store:
        print("\n✅ Connected!")

        print("\n🧪 Running verification...")
        success = verify_connection(vector_store)

    if success:
        print("\n✅ Supabase pgvector is ready for production!")
    else:
        print("\n❌ Verification failed - check your connection")

if __name__ == "__main__":
    main()
