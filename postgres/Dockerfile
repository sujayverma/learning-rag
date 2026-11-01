# PostgreSQL with pgvector extension
# Pin versions for reproducible builds; change PG_MAJOR / PGVECTOR_VERSION as needed.
ARG PG_MAJOR=16

FROM postgres:${PG_MAJOR}

ARG PG_MAJOR
ARG PGVECTOR_VERSION=v0.8.0

# Build pgvector from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        postgresql-server-dev-${PG_MAJOR} \
    && git clone --branch ${PGVECTOR_VERSION} --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/pgvector \
    && apt-get purge -y --auto-remove build-essential git postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

# Auto-enable the extension on first database initialisation
RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/01-pgvector.sql

EXPOSE 5432
