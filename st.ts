import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { resolve, basename } from "node:path";

const EMBED_MODEL = "embeddinggemma";
const EXPAND_MODEL = "qwen3:0.6b";
const RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
const RRF_K = 60;

async function expandQuery(query: string): Promise<string[]> {
  try {
    const res = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({
        model: EXPAND_MODEL,
        prompt: `Generate 3 alternative search queries for: "${query}"
Return only the queries, one per line, no numbers or explanations.`,
        stream: false,
      }),
    });

    if (!res.ok) {
      console.error(`Expand error: ${res.status}`);
      return [query];
    }

    const { response } = (await res.json()) as { response?: string };
    if (!response) return [query];

    // Parse lines, filter empty, remove numbering/bullets
    const expanded = response
      .split("\n")
      .map((l) => l.replace(/^[\d.\-\*\)]+\s*/, "").trim())
      .filter((l) => l.length > 0)
      .slice(0, 3);

    return [query, ...expanded];
  } catch {
    return [query];
  }
}

function progress(current: number, total: number, label: string) {
  const pct = Math.round((current / total) * 100);
  process.stdout.write(`\r\x1b[K${label}: ${current}/${total} (${pct}%)`);
}

function progressDone() {
  process.stdout.write("\n");
}

function getDb() {
  const db = new Database(".cache/search.db");
  sqliteVec.load(db);
  db.run("PRAGMA journal_mode = WAL");

  // Collections table - groups of indexed documents
  db.run(`
    CREATE TABLE IF NOT EXISTS collections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      path TEXT NOT NULL,
      created_at INTEGER DEFAULT (unixepoch()),
      updated_at INTEGER DEFAULT (unixepoch())
    )
  `);

  // Documents table - stores document metadata
  db.run(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      path TEXT NOT NULL,
      content TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at INTEGER DEFAULT (unixepoch()),
      FOREIGN KEY (collection_id) REFERENCES collections(id),
      UNIQUE(collection_id, path)
    )
  `);

  // FTS5 virtual table for full-text search (BM25)
  // name column first for 10x weight in bm25()
  db.run(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      name,
      content,
      content_rowid='id',
      tokenize='porter'
    )
  `);

  // Vector table for semantic search embeddings
  db.run(`
    CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings USING vec0(
      document_id INTEGER PRIMARY KEY,
      hash TEXT,
      model TEXT,
      embedding FLOAT[768]
    )
  `);

  return db;
}

function dropCollection(name: string) {
  const db = getDb();
  const col = db
    .query<
      { id: number },
      [string]
    >(`SELECT id FROM collections WHERE name = ?`)
    .get(name);
  if (!col) return false;

  db.run(
    `DELETE FROM documents_fts WHERE rowid IN (SELECT id FROM documents WHERE collection_id = ?)`,
    [col.id],
  );
  db.run(
    `DELETE FROM document_embeddings WHERE document_id IN (SELECT id FROM documents WHERE collection_id = ?)`,
    [col.id],
  );
  db.run(`DELETE FROM documents WHERE collection_id = ?`, [col.id]);
  db.run(`DELETE FROM collections WHERE id = ?`, [col.id]);
  return true;
}

async function indexFiles(indexPath: string, drop = false) {
  const db = getDb();
  const absolutePath = resolve(indexPath);
  const collectionName = basename(absolutePath);

  if (drop) {
    dropCollection(collectionName);
    console.log(`Dropped collection "${collectionName}"`);
    return;
  }
  db.run(`INSERT OR IGNORE INTO collections (name, path) VALUES (?, ?)`, [
    collectionName,
    absolutePath,
  ]);
  const collection = db
    .query<
      { id: number },
      [string]
    >(`SELECT id FROM collections WHERE name = ?`)
    .get(collectionName);

  if (!collection) {
    console.error("Failed to create collection");
    return;
  }

  // Discover text files
  const glob = new Bun.Glob("**/*.{txt,md}");
  const files = [...glob.scanSync({ cwd: absolutePath, absolute: true })];

  let indexed = 0;
  const indexedNames = new Set<string>();
  for (const filePath of files) {
    // Skip duplicate filenames
    const fileName = basename(filePath);
    if (indexedNames.has(fileName)) continue;
    indexedNames.add(fileName);

    const file = Bun.file(filePath);
    const content = await file.text();
    const hash = Bun.hash(content).toString(16);
    const relativePath = filePath
      .replace(absolutePath, "")
      .replace(/^[\/\\]/, "");

    // Check if already indexed with same hash
    const existing = db
      .query<
        { hash: string },
        [number, string]
      >(`SELECT hash FROM documents WHERE collection_id = ? AND path = ?`)
      .get(collection.id, relativePath);

    if (existing?.hash === hash) {
      continue; // Skip unchanged files
    }

    // Insert or update document
    if (existing) {
      db.run(
        `UPDATE documents SET content = ?, hash = ? WHERE collection_id = ? AND path = ?`,
        [content, hash, collection.id, relativePath],
      );
    } else {
      db.run(
        `INSERT INTO documents (collection_id, name, path, content, hash) VALUES (?, ?, ?, ?, ?)`,
        [collection.id, basename(filePath), relativePath, content, hash],
      );
    }

    // Get document ID and update FTS index
    const doc = db
      .query<
        { id: number },
        [number, string]
      >(`SELECT id FROM documents WHERE collection_id = ? AND path = ?`)
      .get(collection.id, relativePath);

    if (doc) {
      db.run(
        `INSERT OR REPLACE INTO documents_fts (rowid, name, content) VALUES (?, ?, ?)`,
        [doc.id, basename(filePath), content],
      );
    }

    indexed++;
    progress(indexed, files.length, "Indexing");
  }
  if (indexed > 0) progressDone();

  // Remove deleted files from collection
  const allDocs = db
    .query<
      { id: number; path: string },
      [number]
    >(`SELECT id, path FROM documents WHERE collection_id = ?`)
    .all(collection.id);

  let removed = 0;
  for (const doc of allDocs) {
    const fullPath = `${absolutePath}/${doc.path}`;
    const exists = await Bun.file(fullPath).exists();

    if (!exists) {
      db.run(`DELETE FROM documents_fts WHERE rowid = ?`, [doc.id]);
      db.run(`DELETE FROM document_embeddings WHERE document_id = ?`, [doc.id]);
      db.run(`DELETE FROM documents WHERE id = ?`, [doc.id]);
      removed++;
      console.log(`Removed: ${doc.path}`);
    }
  }

  console.log(
    `\nIndexed ${indexed} files, removed ${removed} files in collection "${collectionName}"`,
  );
}

function listCollections() {
  const db = getDb();
  const collections = db
    .query<
      {
        id: number;
        name: string;
        path: string;
        created_at: number;
        doc_count: number;
      },
      []
    >(
      `SELECT c.id, c.name, c.path, c.created_at, COUNT(d.id) as doc_count
     FROM collections c LEFT JOIN documents d ON c.id = d.collection_id
     GROUP BY c.id ORDER BY c.created_at DESC`,
    )
    .all();

  if (!collections.length) {
    console.log("No collections. Use 'index <path>' to create one.");
    return;
  }

  for (const c of collections) {
    console.log(`${c.name} (${c.doc_count} docs) - ${c.path}`);
  }
}

function formatForEmbedding(
  text: string,
  type: "query" | "document",
  title?: string,
): string {
  if (type === "query") {
    return `task: search result | query: ${text}`;
  }
  return `title: ${title || "none"} | text: ${text}`;
}

async function getEmbedding(
  text: string,
  type: "query" | "document",
  title?: string,
): Promise<Float32Array | null> {
  const prompt = formatForEmbedding(text, type, title);
  const res = await fetch("http://localhost:11434/api/embed", {
    method: "POST",
    body: JSON.stringify({ model: EMBED_MODEL, input: prompt }),
  });

  if (!res.ok) {
    console.error(`Ollama error: ${res.status} ${await res.text()}`);
    return null;
  }

  const { embeddings } = (await res.json()) as { embeddings: number[][] };
  if (!embeddings?.[0]) return null;
  return new Float32Array(embeddings[0]);
}

async function embedDocuments(force = false) {
  const db = getDb();

  if (force) {
    db.run(`DELETE FROM document_embeddings`);
    console.log("Cleared all embeddings.");
  }

  // Get docs needing embedding
  const docs = db
    .query<
      { id: number; name: string; content: string; hash: string },
      [string]
    >(
      `SELECT d.id, d.name, d.content, d.hash FROM documents d
     LEFT JOIN document_embeddings e ON d.id = e.document_id
     WHERE e.document_id IS NULL OR e.hash != d.hash OR e.model != ?`,
    )
    .all(EMBED_MODEL);

  if (!docs.length) {
    console.log("All documents already embedded.");
    return;
  }

  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i]!;
    const embedding = await getEmbedding(doc.content, "document", doc.name);
    if (!embedding) {
      console.error(`Failed to embed doc ${doc.id}`);
      continue;
    }

    db.run(
      `INSERT OR REPLACE INTO document_embeddings (document_id, hash, model, embedding) VALUES (?, ?, ?, ?)`,
      [doc.id, doc.hash, EMBED_MODEL, embedding],
    );
    progress(i + 1, docs.length, "Embedding");
  }
  progressDone();

  console.log(`Embedded ${docs.length} documents.`);
}

async function updateAll() {
  const db = getDb();
  const collections = db
    .query<{ path: string }, []>(`SELECT path FROM collections`)
    .all();

  if (!collections.length) {
    console.log("No collections to update.");
    return;
  }

  for (const c of collections) {
    console.log(`\nUpdating: ${c.path}`);
    await indexFiles(c.path);
  }
}

function buildSearchQuery(query: string): string | null {
  const trimmed = query.trim();
  if (trimmed.length < 2) return null;

  // Escape quotes for FTS5 (double them)
  const escaped = trimmed.replace(/"/g, '""');
  const terms = escaped.split(/\s+/).filter((t) => t.length > 1);
  if (!terms.length) return null;

  if (terms.length === 1) {
    return terms[0]!;
  }

  // Multiple terms: "exact phrase" OR (term1 NEAR term2) OR term1 OR term2
  const parts = [
    `"${escaped}"`,
    `(${terms.join(" NEAR ")})`,
    terms.join(" OR "),
  ];
  return parts.join(" OR ");
}

type SearchResult = {
  id: number;
  name: string;
  path: string;
  content: string;
  score: number;
};

// Normalize BM25 score to 0-1 range using sigmoid
function normalizeBM25(score: number): number {
  // BM25 scores are negative in SQLite (lower = better)
  // Typical range: -15 (excellent) to -2 (weak match)
  // Map to 0-1 where higher is better
  const absScore = Math.abs(score);
  // Sigmoid-ish normalization: maps ~2-15 range to ~0.1-0.95
  return 1 / (1 + Math.exp(-(absScore - 5) / 3));
}

function searchBM25(query: string, limit = 5): SearchResult[] {
  const ftsQuery = buildSearchQuery(query);
  if (!ftsQuery) return [];

  const db = getDb();
  try {
    // Fetch extra results to account for duplicates
    // bm25 weights: name=10x, content=1x
    const results = db
      .query<SearchResult, [string, number]>(
        `SELECT d.id, d.name, d.path, d.content, bm25(documents_fts, 10.0, 1.0) as score
         FROM documents_fts f
         JOIN documents d ON f.rowid = d.id
         WHERE documents_fts MATCH ?
         ORDER BY bm25(documents_fts, 10.0, 1.0)
         LIMIT ?`,
      )
      .all(ftsQuery, limit);

    if (!results.length) return [];

    // Normalize scores
    return results.map((r) => ({ ...r, score: normalizeBM25(r.score) }));
  } catch {
    return [];
  }
}

async function searchVector(
  query: string,
  limit = 5,
  debug = false,
): Promise<SearchResult[]> {
  const db = getDb();
  const queries = await expandQuery(query);

  if (debug) {
    console.log("Queries:", queries);
  }

  // Run search for each query, keep highest score per document
  const docScores = new Map<number, number>();

  for (const q of queries) {
    const embedding = await getEmbedding(q, "query");
    if (!embedding) continue;

    const results = db
      .query<{ document_id: number; distance: number }, [Float32Array, number]>(
        `SELECT document_id, distance
         FROM document_embeddings
         WHERE embedding MATCH ?
         ORDER BY distance
         LIMIT ?`,
      )
      .all(embedding, limit * 2);

    for (const r of results) {
      const score = 1 / (r.distance + 1);
      const existing = docScores.get(r.document_id) || 0;
      if (score > existing) {
        docScores.set(r.document_id, score);
      }
    }
  }

  if (!docScores.size) return [];

  // Sort by score and take top limit
  const topDocs = [...docScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);

  // Only fetch details for top results
  const docIds = topDocs.map(([id]) => id);
  const docs = db
    .query<
      { id: number; name: string; path: string; content: string },
      []
    >(`SELECT id, name, path, content FROM documents WHERE id IN (${docIds.join(",")})`)
    .all();

  const docMap = new Map(docs.map((d) => [d.id, d]));

  return topDocs
    .map(([id, score]) => {
      const doc = docMap.get(id);
      if (!doc) return null;
      return { ...doc, score };
    })
    .filter((r): r is SearchResult => r !== null);
}

async function searchVectorSingle(
  query: string,
  limit: number,
): Promise<SearchResult[]> {
  const embedding = await getEmbedding(query, "query");
  if (!embedding) return [];

  const db = getDb();
  const results = db
    .query<{ document_id: number; distance: number }, [Float32Array, number]>(
      `SELECT document_id, distance FROM document_embeddings
       WHERE embedding MATCH ? ORDER BY distance LIMIT ?`,
    )
    .all(embedding, limit);

  if (!results.length) return [];

  const docIds = results.map((r) => r.document_id);
  const docs = db
    .query<
      { id: number; name: string; path: string; content: string },
      []
    >(`SELECT id, name, path, content FROM documents WHERE id IN (${docIds.join(",")})`)
    .all();

  const docMap = new Map(docs.map((d) => [d.id, d]));

  return results
    .map((r) => {
      const doc = docMap.get(r.document_id);
      if (!doc) return null;
      return { ...doc, score: 1 / (r.distance + 1) };
    })
    .filter((r): r is SearchResult => r !== null);
}

function rrfFusion(
  resultSets: { results: SearchResult[]; weight: number }[],
): Map<number, { doc: SearchResult; score: number }> {
  const scores = new Map<number, { doc: SearchResult; score: number }>();

  for (const { results, weight } of resultSets) {
    for (let rank = 0; rank < results.length; rank++) {
      const doc = results[rank]!;
      const rrfScore = weight / (rank + 1 + RRF_K);
      const existing = scores.get(doc.id);
      if (existing) {
        existing.score += rrfScore;
      } else {
        scores.set(doc.id, { doc, score: rrfScore });
      }
    }
  }

  return scores;
}

const RERANK_SYSTEM = `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`;

function buildRerankPrompt(
  query: string,
  title: string,
  content: string,
): string {
  const userPrompt = `
<Instruct>: Determine if the document is relevant to the search query. atch documents that discuss the queried topic, even if phrasing differs.
<Query>: ${query}
<Document Title>: ${title}
<Document>: ${content.slice(0, 4000)}`;

  return `<|im_start|>system
${RERANK_SYSTEM}<|im_end|>
<|im_start|>user
${userPrompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

`;
}

function logprobToConfidence(logprob: number): number {
  // Convert log probability to probability (0-1)
  // logprob is typically negative, closer to 0 = higher confidence
  const prob = Math.exp(logprob);
  return Math.min(1, Math.max(0, prob));
}

function computeRerankScore(logprobs: LogProb[]): number {
  // Find "yes" or "no" token and use its logprob for confidence
  // Returns: 0-1 score

  let yesIdx = -1;
  let noIdx = -1;
  let yesLogprob = 0;
  let noLogprob = 0;

  for (let i = 0; i < logprobs.length; i++) {
    const token = logprobs[i]!.token.toLowerCase();
    if (yesIdx === -1 && token.includes("yes")) {
      yesIdx = i;
      yesLogprob = logprobs[i]!.logprob;
    }
    if (noIdx === -1 && token.includes("no")) {
      noIdx = i;
      noLogprob = logprobs[i]!.logprob;
    }
  }

  // Pick whichever appears first (-1 means not found)
  const isYes = yesIdx !== -1 && (noIdx === -1 || yesIdx < noIdx);
  const isNo = noIdx !== -1 && (yesIdx === -1 || noIdx < yesIdx);

  if (isYes) {
    const confidence = logprobToConfidence(yesLogprob);
    // yes with high confidence → 1, yes with low confidence → 0.6
    return 0.6 + confidence * 0.4;
  } else if (isNo) {
    const confidence = logprobToConfidence(noLogprob);
    // no with high confidence → 0, no with low confidence → 0.4
    return 0.4 - confidence * 0.4;
  }
  // Unknown answer → neutral score
  return 0.5;
}

type LogProb = { token: string; logprob: number };
type RerankResponse = {
  response: string;
  logprobs?: LogProb[];
};

async function rerankSingleDoc(
  query: string,
  doc: SearchResult,
): Promise<{ id: number; score: number }> {
  const prompt = buildRerankPrompt(query, doc.name, doc.content);

  try {
    const res = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({
        model: RERANK_MODEL,
        prompt,
        stream: false,
        raw: true,
        options: {
          num_predict: 2,
        },
        logprobs: true,
      }),
    });

    if (!res.ok) {
      return { id: doc.id, score: 5 };
    }

    const data = (await res.json()) as RerankResponse;

    const score = computeRerankScore(data.logprobs ?? []);
    return { id: doc.id, score };
  } catch {
    return { id: doc.id, score: 5 };
  }
}

const RERANK_CONCURRENCY = 5;

async function rerankWithLLM(
  query: string,
  docs: SearchResult[],
): Promise<Map<number, number>> {
  const scores = new Map<number, number>();

  // Process in parallel batches
  for (let i = 0; i < docs.length; i += RERANK_CONCURRENCY) {
    const batch = docs.slice(i, i + RERANK_CONCURRENCY);
    const results = await Promise.all(
      batch.map((doc) => rerankSingleDoc(query, doc)),
    );
    for (const { id, score } of results) {
      scores.set(id, score);
    }
  }

  return scores;
}

function positionBlend(rrfRank: number, rerankerScore: number): number {
  const rrfScore = 1 / rrfRank; // Position-based: 1, 0.5, 0.33...

  let retrieverWeight: number;
  if (rrfRank <= 3) {
    retrieverWeight = 0.75;
  } else if (rrfRank <= 10) {
    retrieverWeight = 0.6;
  } else {
    retrieverWeight = 0.4;
  }

  return retrieverWeight * rrfScore + (1 - retrieverWeight) * rerankerScore;
}

async function searchCombined(
  query: string,
  limit = 5,
  debug = false,
): Promise<SearchResult[]> {
  const queries = await expandQuery(query);
  const original = queries[0]!;
  const expanded = queries.slice(1);

  if (debug) console.log("Queries:", queries);

  // Original query gets higher weight
  const bm25Original = searchBM25(original, 50);
  const vecOriginal = await searchVectorSingle(original, 50);

  // Each expanded query as separate result set for proper ranking
  const bm25Expanded = expanded.map((q) => searchBM25(q, 30));
  const vecExpanded = await Promise.all(
    expanded.map((q) => searchVectorSingle(q, 30)),
  );

  const rrfScores = rrfFusion([
    { results: bm25Original, weight: 2 },
    { results: vecOriginal, weight: 2 },
    ...bm25Expanded.map((results) => ({ results, weight: 1 })),
    ...vecExpanded.map((results) => ({ results, weight: 1 })),
  ]);

  const sorted = [...rrfScores.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, 30);

  if (debug) console.log("RRF top:", sorted.length);

  const rerankerScores = await rerankWithLLM(
    original,
    sorted.map((s) => s.doc),
  );

  const final = sorted.map((s, idx) => {
    const rerankerScore = rerankerScores.get(s.doc.id) || 0.5;
    const blendedScore = positionBlend(idx + 1, rerankerScore);
    return { ...s.doc, score: blendedScore };
  });

  return final.sort((a, b) => b.score - a.score).slice(0, limit);
}

const YELLOW = "\x1b[33m";
const RESET = "\x1b[0m";

function highlightMatches(text: string, queryWords: string[]): string {
  // Build regex to match any query word (case insensitive)
  const escaped = queryWords
    .filter((w) => w.length > 1)
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!escaped.length) return text;

  const regex = new RegExp(`(${escaped.join("|")})`, "gi");
  return text.replace(regex, `${YELLOW}$1${RESET}`);
}

function formatResult(r: SearchResult, query: string, idx: number): string {
  const queryWords = query.toLowerCase().split(/\s+/);
  const lines = r.content.split("\n").filter((l) => l.trim());

  let bestIdx = 0;
  let bestCount = 0;

  for (let i = 0; i < lines.length; i++) {
    const lineLower = lines[i]!.toLowerCase();
    const count = queryWords.filter((w) => lineLower.includes(w)).length;
    if (count > bestCount) {
      bestCount = count;
      bestIdx = i;
    }
  }

  const snippetLines: string[] = [];
  if (bestIdx > 0) snippetLines.push(lines[bestIdx - 1]!);
  snippetLines.push(lines[bestIdx]!);
  if (bestIdx < lines.length - 1) snippetLines.push(lines[bestIdx + 1]!);

  const snippet = snippetLines
    .map((l) => l.trim())
    .join(" ")
    .slice(0, 100);

  const highlightedSnippet = highlightMatches(snippet, queryWords);
  const highlightedName = highlightMatches(r.name, queryWords);
  return `${idx}. ${highlightedName} [${r.score.toFixed(2)}]\n   ${highlightedSnippet}...`;
}

const rawArgs = process.argv.slice(2);
const cmd = rawArgs[0];

// Todo usage instructions

if (cmd === "index") {
  const drop = rawArgs.includes("--drop");
  const path = rawArgs.slice(1).find((a) => !a.startsWith("--")) ?? ".";
  await indexFiles(path, drop);
} else if (cmd === "collections") {
  listCollections();
} else if (cmd === "update") {
  await updateAll();
} else if (cmd === "embed") {
  await embedDocuments(rawArgs.includes("--force"));
} else if (cmd === "search") {
  const query = rawArgs.slice(1).join(" ");
  if (!query || query.trim().length < 2) {
    console.log("Query must be at least 2 characters.");
    process.exit(1);
  }
  const results = searchBM25(query);
  if (!results.length) {
    console.log("No results found.");
    process.exit(0);
  }
  for (let i = 0; i < results.length; i++) {
    if (i > 0) console.log();
    console.log(formatResult(results[i]!, query, i + 1));
  }
} else if (cmd === "vsearch") {
  const debug = rawArgs.includes("--debug");
  const query = rawArgs
    .slice(1)
    .filter((a) => !a.startsWith("--"))
    .join(" ");
  if (!query || query.trim().length < 2) {
    console.log("Query must be at least 2 characters.");
    process.exit(1);
  }
  const results = await searchVector(query, 5, debug);
  if (!results.length) {
    console.log("No results found.");
    process.exit(0);
  }
  for (let i = 0; i < results.length; i++) {
    if (i > 0) console.log();
    console.log(formatResult(results[i]!, query, i + 1));
  }
} else if (cmd === "query") {
  const debug = rawArgs.includes("--debug");
  const query = rawArgs
    .slice(1)
    .filter((a) => !a.startsWith("--"))
    .join(" ");
  if (!query || query.trim().length < 2) {
    console.log("Query must be at least 2 characters.");
    process.exit(1);
  }
  const results = await searchCombined(query, 5, debug);
  if (!results.length) {
    console.log("No results found.");
    process.exit(0);
  }
  for (let i = 0; i < results.length; i++) {
    if (i > 0) console.log();
    console.log(formatResult(results[i]!, query, i + 1));
  }
} else {
  console.log(`Usage: st <command>

Commands:
  index <path> [--drop]  Index files in a directory (default: .)
  collections            List all collections
  update                 Re-index all collections
  embed [--force]        Generate embeddings for indexed documents
  search <query>         Search indexed documents (BM25)
  vsearch <query> [--debug]  Search indexed documents (vector)
  query <query> [--debug]    Combined search (BM25 + vector + rerank)`);
  if (cmd) console.log(`\nUnknown command: ${cmd}`);
}
