import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { resolve, basename } from "node:path";

const EMBED_MODEL = "embeddinggemma";

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
  db.run(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
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
  for (const filePath of files) {
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
        `INSERT OR REPLACE INTO documents_fts (rowid, content) VALUES (?, ?)`,
        [doc.id, content],
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

function searchBM25(query: string, limit = 5): SearchResult[] {
  const ftsQuery = buildSearchQuery(query);
  if (!ftsQuery) return [];

  const db = getDb();
  try {
    return db
      .query<SearchResult, [string, number]>(
        `SELECT d.id, d.name, d.path, d.content, abs(f.rank) as score
         FROM documents_fts f
         JOIN documents d ON f.rowid = d.id
         WHERE documents_fts MATCH ?
         ORDER BY f.rank
         LIMIT ?`,
      )
      .all(ftsQuery, limit);
  } catch {
    return [];
  }
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
  for (const r of results) {
    console.log(`[${r.score.toFixed(2)}] ${r.path}`);
    console.log(`  ${r.content.slice(0, 100).replace(/\n/g, " ")}...`);
  }
} else {
  console.log(`Usage: st <command>

Commands:
  index <path> [--drop]  Index files in a directory (default: .)
  collections            List all collections
  update                 Re-index all collections
  embed [--force]        Generate embeddings for indexed documents
  search <query>         Search indexed documents`);
  if (cmd) console.log(`\nUnknown command: ${cmd}`);
}
