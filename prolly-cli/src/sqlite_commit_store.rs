//! SQLite-based commit graph store for the CLI.
//!
//! This is separate from the core library to maintain WASM compatibility.

use prolly_core::{Commit, CommitGraphStore, Hash};
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
    fn put_commit(&self, commit_hash: &Hash, commit: Commit) {
        let conn = self.conn.lock().unwrap();

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
        .expect("Failed to insert commit");

        // Delete existing parents
        conn.execute(
            "DELETE FROM commit_parents WHERE commit_hash = ?1",
            params![commit_hash],
        )
        .expect("Failed to delete old parents");

        // Insert parents
        for (idx, parent_hash) in commit.parents.iter().enumerate() {
            conn.execute(
                "INSERT INTO commit_parents (commit_hash, parent_hash, parent_index)
                 VALUES (?1, ?2, ?3)",
                params![commit_hash, parent_hash, idx as i64],
            )
            .expect("Failed to insert parent");
        }
    }

    fn get_commit(&self, commit_hash: &Hash) -> Option<Commit> {
        let conn = self.conn.lock().unwrap();

        // Get commit data
        let mut stmt = conn
            .prepare(
                "SELECT tree_root, message, timestamp, author, pattern, seed FROM commits WHERE hash = ?1",
            )
            .ok()?;

        let commit = stmt
            .query_row(params![commit_hash], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })
            .ok()?;

        // Get parents
        let mut stmt = conn
            .prepare(
                "SELECT parent_hash FROM commit_parents WHERE commit_hash = ?1 ORDER BY parent_index",
            )
            .ok()?;

        let parents = stmt
            .query_map(params![commit_hash], |row| row.get::<_, Vec<u8>>(0))
            .ok()?
            .filter_map(Result::ok)
            .collect();

        Some(Commit {
            tree_root: commit.0,
            parents,
            message: commit.1,
            timestamp: commit.2,
            author: commit.3,
            pattern: commit.4,
            seed: commit.5 as u32,
        })
    }

    fn get_parents(&self, commit_hash: &Hash) -> Vec<Hash> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT parent_hash FROM commit_parents WHERE commit_hash = ?1 ORDER BY parent_index",
            )
            .expect("Failed to prepare statement");

        stmt.query_map(params![commit_hash], |row| row.get::<_, Vec<u8>>(0))
            .expect("Failed to query parents")
            .filter_map(Result::ok)
            .collect()
    }

    fn set_ref(&self, name: &str, commit_hash: &Hash) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO refs (name, commit_hash) VALUES (?1, ?2)",
            params![name, commit_hash],
        )
        .expect("Failed to set ref");
    }

    fn get_ref(&self, name: &str) -> Option<Hash> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT commit_hash FROM refs WHERE name = ?1")
            .ok()?;

        stmt.query_row(params![name], |row| row.get::<_, Vec<u8>>(0))
            .ok()
    }

    fn list_refs(&self) -> HashMap<String, Hash> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT name, commit_hash FROM refs")
            .expect("Failed to prepare statement");

        stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .expect("Failed to query refs")
        .filter_map(Result::ok)
        .collect()
    }

    fn set_head(&self, ref_name: &str) {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('HEAD', ?1)",
            params![ref_name],
        )
        .expect("Failed to set HEAD");
    }

    fn get_head(&self) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT value FROM metadata WHERE key = 'HEAD'")
            .ok()?;

        stmt.query_row([], |row| row.get::<_, String>(0)).ok()
    }

    fn find_commit_by_prefix(&self, prefix: &str) -> Option<Hash> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("{}%", prefix.to_uppercase());

        let mut stmt = conn
            .prepare("SELECT hash FROM commits WHERE hex(hash) LIKE ?1")
            .ok()?;

        let matches: Vec<Hash> = stmt
            .query_map(params![pattern], |row| row.get::<_, Vec<u8>>(0))
            .ok()?
            .filter_map(Result::ok)
            .collect();

        if matches.len() == 1 {
            Some(matches[0].clone())
        } else {
            None
        }
    }
}
