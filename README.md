# Search tool

Cli tool for searching text documents using: BM25, vector semantic search and LLM re-ranking.


## Architecture

1. User query
2. Query expansion with fast LLM
3. Continue with original query and expanded query with smaller weight
4. Run BM25
5. Run vector semantic search
6. Combine results with weights, keep top k results
7. Rerank results with LLM


### Search backends


- BM25 (sqlite3) | abs(score)
- vector Cosine distance | 1 / (distance + 1)
- LLM re-ranking | 0-10 rating


### Combining strategy


Reciprocal Rank Fusion (RRF) with position aware blending

- original query 2x weight, expanded query 1x weight
- full text search and vector index for both queries
- combine reults with RRF 
RRF formula:
- `sum over result sets (1 / (rank_i + k))`
- k smoothing factor set to 50
- Take top 30 results for reranking
- LLM scores each result (yes/no), convert logprob to score (0-10)
- Position aware blending:

Position-Aware Blending:
   - RRF rank 1-3: 75% retrieval, 25% reranker (preserves exact matches)
   - RRF rank 4-10: 60% retrieval, 40% reranker
   - RRF rank 11+: 40% retrieval, 60% reranker (trust reranker more)

## Requirements

- **Bun** >= 1.0.0
- **SQLite** >= 3.44.0
- **Ollama** locally installed

### Ollama models

- `embeddinggemma3` | Vector embedding model | 1.6GB
- `ExpedientFalcon/qwen3-reranker:0.6b-q8_0` | Re-ranking (trained) | ~640MB
- `qwen3:0.6b` | Query expansion | ~400MB


## Installation

```sh
bun install
```

## Indexing

File -> FTS5 index -> SQLite database

## Embedding


Document -> Format for EmbeddingGemma -> EmbeddingGemma -> Vector

## Searching

Query -> Query expansion -> BM25 + vector semantic search -> RRF -> LLM re-ranking -> Results