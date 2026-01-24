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

- BM25 (sqlite3) | normalized score
- vector Cosine distance | 1 / (distance + 1)
- LLM re-ranking | 0-1 confidence score

### Combining strategy

Reciprocal Rank Fusion (RRF) with position aware blending

- original query 2x weight, expanded query 1x weight
- full text search and vector index for both queries
- combine results with RRF, normalize to 0-1
  RRF formula:
- `sum over result sets (1 / (rank_i + k))`
- k smoothing factor set to 60
- Take top 30 results for reranking
- LLM scores each result (yes/no), convert logprob to confidence (0-1)
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

```sh
ollama pull embeddinggemma
ollama pull ExpedientFalcon/qwen3-reranker:0.6b-q8_0
ollama pull qwen3:0.6b
```

## Installation

```sh
bun install
bun link    # Makes 'st' command available globally
```

## Usage

### Index documents

Index a directory of text files (`.txt`, `.md`):

```sh
st index <path>
```

Re-index and drop existing collection:

```sh
st index <path> --drop
```

### Manage collections

List all indexed collections:

```sh
st collections
```

Re-index all existing collections:

```sh
st update
```

### Generate embeddings

Generate vector embeddings for all indexed documents:

```sh
st embed
```

Force re-embed all documents:

```sh
st embed --force
```

### Search

**BM25 search** (fast, keyword-based):

```sh
st search <query>
```

**Vector search** (semantic similarity):

```sh
st vsearch <query>
st vsearch <query> --debug  # Show expanded queries
```

**Combined search** (BM25 + vector + LLM reranking):

```sh
st query <query>
st query <query> --debug  # Show debug info
```

### Example workflow

```sh
# 1. Index your documents
st index ./my-documents

# 2. Generate embeddings
st embed

# 3. Search
st query "how to configure authentication"
```

## How it works

### Indexing

File -> FTS5 index -> SQLite database

### Embedding

Document -> Format for EmbeddingGemma -> EmbeddingGemma -> Vector

### Searching

Query -> Query expansion -> BM25 + vector semantic search -> RRF -> LLM re-ranking -> Results
