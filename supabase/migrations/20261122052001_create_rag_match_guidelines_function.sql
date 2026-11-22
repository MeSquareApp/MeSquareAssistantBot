-- Create RPC match function for vector similarity search using Qwen3-Embedding-4B (2560 dims)
CREATE OR REPLACE FUNCTION rag.match_guidelines (
    query_embedding halfvec(2560),
    match_threshold float DEFAULT 0.70,
    match_count int DEFAULT 4
)
RETURNS TABLE (
    id UUID,
    chunk_id TEXT,
    guideline_title TEXT,
    publication_year INT,
    section_title TEXT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        id,
        chunk_id,
        guideline_title,
        publication_year,
        section_title,
        content,
        1 - (embedding <=> query_embedding) AS similarity
    FROM rag.guideline_chunks
    WHERE status = 'active'
      AND 1 - (embedding <=> query_embedding) >= match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
