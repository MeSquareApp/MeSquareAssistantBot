-- 1. Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create dedicated schema
CREATE SCHEMA IF NOT EXISTS rag;

-- 3. Create guideline chunks table with halfvec(2560) for Qwen3-Embedding-4B
DROP TABLE IF EXISTS rag.guideline_chunks CASCADE;

CREATE TABLE rag.guideline_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id TEXT UNIQUE NOT NULL,
    guideline_title TEXT NOT NULL,
    publication_year INT NOT NULL,
    section_title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding halfvec(2560) NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Create HNSW index for cosine similarity search (halfvec supports up to 4096 dimensions)
CREATE INDEX IF NOT EXISTS idx_guideline_chunks_embedding 
ON rag.guideline_chunks 
USING hnsw (embedding halfvec_cosine_ops);

-- 5. Create GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_guideline_chunks_fts 
ON rag.guideline_chunks 
USING gin (to_tsvector('english', content));

-- 6. Enable Row Level Security and allow read access
ALTER TABLE rag.guideline_chunks ENABLE ROW LEVEL SECURITY;

DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'guideline_chunks' 
        AND policyname = 'Allow read access to all users'
    ) THEN
        CREATE POLICY "Allow read access to all users" 
        ON rag.guideline_chunks 
        FOR SELECT 
        USING (true);
    END IF;
END $$;
