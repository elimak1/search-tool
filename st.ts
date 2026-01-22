import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import { resolve, basename } from "node:path";

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
  // embeddinggemma3 produces 768-dimensional vectors
  db.run(`
    CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings USING vec0(
      document_id INTEGER PRIMARY KEY,
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
  const files = glob.scanSync({ cwd: absolutePath, absolute: true });

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
        `INSERT INTO documents (collection_id, path, content, hash) VALUES (?, ?, ?, ?)`,
        [collection.id, relativePath, content, hash],
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
    console.log(`Indexed: ${relativePath}`);
  }

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
} else {
  console.log(`Usage: st <command>

Commands:
  index <path> [--drop]  Index files in a directory (default: .)
  collections            List all collections
  update                 Re-index all collections`);
  if (cmd) console.log(`\nUnknown command: ${cmd}`);
}
