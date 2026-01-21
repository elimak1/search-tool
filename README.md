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

## Usage