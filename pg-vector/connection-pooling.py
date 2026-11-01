"""Learning connection pooling to PostgreSQL using psycopg2 and pgvector."""

import os
from contextlib import contextmanager
from openai import OpenAI
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

env_name = os.getenv("ENV_NAME", "local")
print(f"Connecting to {env_name} database...")

if env_name == "local":
    db_url = os.getenv("LOCAL_DB_URL")
elif env_name == "dev":
    db_url = os.getenv("SUPABASE_DB_URL")
else:
    raise ValueError(f"Unknown ENV_NAME: {env_name}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DatabaseConnectionPool:
    """Manage a PostgreSQL connection pool using psycopg2."""

    def __init__(self, db_url, minconn=1, maxconn=5):
        self.db_url = db_url
        self.minconn = minconn
        self.maxconn = maxconn
        self.connection_pool = None

    def initialize_pool(self):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            self.minconn, self.maxconn, self.db_url
        )
        print("Connection pool created successfully.")

    @contextmanager
    def get_connection(self):
        if self.connection_pool is None:
            raise RuntimeError("Pool not initialized. Call initialize_pool() first.")
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def close_pool(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            print("All connections in the pool closed.")

    def insert_document(self, content: str, embedding: list):
        """Insert a single document with its embedding. Returns the new row id."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector) RETURNING id;",
                    (content, embedding),
                )
                return cursor.fetchone()[0]

    def create_documents_table(self):
        """Create the documents table (and vector extension) if they don't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id        SERIAL PRIMARY KEY,
                        content   TEXT NOT NULL,
                        embedding VECTOR(1536)
                    );
                """)
        print("documents table ready.")


db_pool = DatabaseConnectionPool(db_url, minconn=2, maxconn=5)


def search_with_pool(query_embedding: list, limit: int = 5):
    """Run a pgvector similarity search using a pooled connection."""
    with db_pool.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            return cursor.fetchall()

def embed(text: str) -> list:
    resp = client.embeddings.create(
        model="text-embedding-3-small",   # 1536 dims — matches VECTOR(1536)
        input=text,
    )
    return resp.data[0].embedding


def semantic_search(query_text: str, limit: int = 5):
    query_vec = embed(query_text)
    return search_with_pool(query_vec, limit)


def demonstrate_pooling():
    """Show connection pooling in action with several queries."""
    print("\nDemonstrating connection pooling with multiple queries...\n")

    db_pool.create_documents_table()

    # in demonstrate_pooling(), instead of fake vectors:
    texts = [
        "PostgreSQL connection pooling reuses database connections",
        "pgvector enables similarity search on embeddings",
        "Supabase provides a shared pooler over IPv4",
    ]
    for t in texts:
        doc_id = db_pool.insert_document(t, embed(t))
        print(f"Inserted id={doc_id}: {t}")

    for content, similarity in semantic_search("connection pooling", limit=3):
        print(f"{similarity:.4f}  {content}")

    for i in range(5):
        with db_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM documents;")
                count = cursor.fetchone()[0]
        print(f"Query {i + 1}: Total documents: {count} (pooled connection)")

    print("\n✅ All queries used pooled connections!")


def main():
    print("=" * 60)
    print("Connection Pooling for Production")
    print("=" * 60)

    db_pool.initialize_pool()
    try:
        demonstrate_pooling()
    except Exception as e:
        print(f"\n⚠️  Could not run demo (need documents table): {e}")
    finally:
        db_pool.close_pool()


if __name__ == "__main__":
    main()