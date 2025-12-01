//! SQLite-based commit graph store for the CLI.
//!
//! This is separate from the core library to maintain WASM compatibility.

use prolly_core::{Commit, CommitGraphStore, Hash, StoreError, StoreResult};
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// SQLite-based implementation of CommitGraphStore
pub struct SqliteCommitGraphStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteCommitGraphStore {
    /// Create a new SQLite commit graph store
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to SQLite database file
    pub fn new<P: AsRef<Path>>(db_path: P) -> anyhow::Result<Self> {
        let conn = Connection::open(db_path)?;
        let store = SqliteCommitGraphStore {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.create_tables()?;
        Ok(store)
    }

    fn create_tables(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();

        // Commits table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS commits (
                hash BLOB PRIMARY KEY,
                tree_root BLOB NOT NULL,
                message TEXT NOT NULL,
                timestamp REAL NOT NULL,
                author TEXT NOT NULL,
                pattern REAL NOT NULL DEFAULT 0.01,
                seed INTEGER NOT NULL DEFAULT 42
            )",
            [],
        )?;

        // Commit parents table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS commit_parents (
                commit_hash BLOB NOT NULL,
                parent_hash BLOB NOT NULL,
                parent_index INTEGER NOT NULL,
                PRIMARY KEY (commit_hash, parent_index),
                FOREIGN KEY (commit_hash) REFERENCES commits(hash)
            )",
            [],
        )?;

        // Refs table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS refs (
                name TEXT PRIMARY KEY,
                commit_hash BLOB NOT NULL,
                FOREIGN KEY (commit_hash) REFERENCES commits(hash)
            )",
            [],
        )?;

        // Metadata table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        Ok(())
    }
}

impl CommitGraphStore for SqliteCommitGraphStore {
    fn put_commit(&self, commit_hash: &Hash, commit: Commit) -> StoreResult<()> {
        let conn = self.conn.lock()?;

        // Insert commit
        conn.execute(
            "INSERT OR REPLACE INTO commits (hash, tree_root, message, timestamp, author, pattern, seed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                commit_hash,
                &commit.tree_root,
                &commit.message,
                commit.timestamp,
                &commit.author,
                commit.pattern,
                commit.seed as i64,
            ],
        )
        .map_err(|e| StoreError::Database(format!("Failed to insert commit: {}", e)))?;

        // Delete existing parents
        conn.execute(
            "DELETE FROM commit_parents WHERE commit_hash = ?1",
            params![commit_hash],
        )
        .map_err(|e| StoreError::Database(format!("Failed to delete old parents: {}", e)))?;

        // Insert parents
        for (idx, parent_hash) in commit.parents.iter().enumerate() {
            conn.execute(
                "INSERT INTO commit_parents (commit_hash, parent_hash, parent_index)
                 VALUES (?1, ?2, ?3)",
                params![commit_hash, parent_hash, idx as i64],
            )
            .map_err(|e| StoreError::Database(format!("Failed to insert parent: {}", e)))?;
        }

        Ok(())
    }

    fn get_commit(&self, commit_hash: &Hash) -> StoreResult<Option<Commit>> {
        let conn = self.conn.lock()?;

        // Get commit data
        let mut stmt = conn
            .prepare(
                "SELECT tree_root, message, timestamp, author, pattern, seed FROM commits WHERE hash = ?1",
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        let commit = match stmt.query_row(params![commit_hash], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        }) {
            Ok(c) => c,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(StoreError::Database(format!("Failed to query commit: {}", e))),
        };

        // Get parents
        let mut stmt = conn
            .prepare(
                "SELECT parent_hash FROM commit_parents WHERE commit_hash = ?1 ORDER BY parent_index",
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        let parents: Vec<Hash> = stmt
            .query_map(params![commit_hash], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|e| StoreError::Database(format!("Failed to query parents: {}", e)))?
            .filter_map(Result::ok)
            .collect();

        Ok(Some(Commit {
            tree_root: commit.0,
            parents,
            message: commit.1,
            timestamp: commit.2,
            author: commit.3,
            pattern: commit.4,
            seed: commit.5 as u32,
        }))
    }

    fn get_parents(&self, commit_hash: &Hash) -> StoreResult<Vec<Hash>> {
        let conn = self.conn.lock()?;

        let mut stmt = conn
            .prepare(
                "SELECT parent_hash FROM commit_parents WHERE commit_hash = ?1 ORDER BY parent_index",
            )
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        let parents: Vec<Hash> = stmt
            .query_map(params![commit_hash], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|e| StoreError::Database(format!("Failed to query parents: {}", e)))?
            .filter_map(Result::ok)
            .collect();

        Ok(parents)
    }

    fn set_ref(&self, name: &str, commit_hash: &Hash) -> StoreResult<()> {
        let conn = self.conn.lock()?;
        conn.execute(
            "INSERT OR REPLACE INTO refs (name, commit_hash) VALUES (?1, ?2)",
            params![name, commit_hash],
        )
        .map_err(|e| StoreError::Database(format!("Failed to set ref: {}", e)))?;
        Ok(())
    }

    fn get_ref(&self, name: &str) -> StoreResult<Option<Hash>> {
        let conn = self.conn.lock()?;
        let mut stmt = conn
            .prepare("SELECT commit_hash FROM refs WHERE name = ?1")
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        match stmt.query_row(params![name], |row| row.get::<_, Vec<u8>>(0)) {
            Ok(hash) => Ok(Some(hash)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StoreError::Database(format!("Failed to get ref: {}", e))),
        }
    }

    fn list_refs(&self) -> StoreResult<HashMap<String, Hash>> {
        let conn = self.conn.lock()?;
        let mut stmt = conn
            .prepare("SELECT name, commit_hash FROM refs")
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        let refs: HashMap<String, Hash> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| StoreError::Database(format!("Failed to query refs: {}", e)))?
            .filter_map(Result::ok)
            .collect();

        Ok(refs)
    }

    fn set_head(&self, ref_name: &str) -> StoreResult<()> {
        let conn = self.conn.lock()?;
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('HEAD', ?1)",
            params![ref_name],
        )
        .map_err(|e| StoreError::Database(format!("Failed to set HEAD: {}", e)))?;
        Ok(())
    }

    fn get_head(&self) -> StoreResult<Option<String>> {
        let conn = self.conn.lock()?;
        let mut stmt = conn
            .prepare("SELECT value FROM metadata WHERE key = 'HEAD'")
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        match stmt.query_row([], |row| row.get::<_, String>(0)) {
            Ok(head) => Ok(Some(head)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StoreError::Database(format!("Failed to get HEAD: {}", e))),
        }
    }

    fn find_commit_by_prefix(&self, prefix: &str) -> StoreResult<Option<Hash>> {
        let conn = self.conn.lock()?;
        let pattern = format!("{}%", prefix.to_uppercase());

        let mut stmt = conn
            .prepare("SELECT hash FROM commits WHERE hex(hash) LIKE ?1")
            .map_err(|e| StoreError::Database(format!("Failed to prepare statement: {}", e)))?;

        let matches: Vec<Hash> = stmt
            .query_map(params![pattern], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|e| StoreError::Database(format!("Failed to query commits: {}", e)))?
            .filter_map(Result::ok)
            .collect();

        if matches.len() == 1 {
            Ok(Some(matches[0].clone()))
        } else {
            Ok(None)
        }
    }
}
