//! Commit graph store implementations for ProllyPy.
//!
//! Provides storage backends for commits and references in the version control system.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::Hash;

/// Represents a commit in the version control system.
///
/// Each commit points to a prolly tree root and zero or more parent commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    /// Hash of the prolly tree root node
    pub tree_root: Hash,
    /// List of parent commit hashes
    pub parents: Vec<Hash>,
    /// Commit message
    pub message: String,
    /// Unix timestamp
    pub timestamp: f64,
    /// Author name/email
    pub author: String,
    /// Split probability for the prolly tree
    #[serde(default = "default_pattern")]
    pub pattern: f64,
    /// Seed for rolling hash function
    #[serde(default = "default_seed")]
    pub seed: u32,
}

fn default_pattern() -> f64 {
    0.01
}

fn default_seed() -> u32 {
    42
}

impl Commit {
    /// Create a new commit
    pub fn new(
        tree_root: Hash,
        parents: Vec<Hash>,
        message: String,
        timestamp: f64,
        author: String,
    ) -> Self {
        Commit {
            tree_root,
            parents,
            message,
            timestamp,
            author,
            pattern: 0.01,
            seed: 42,
        }
    }

    /// Compute the hash of this commit based on its contents
    pub fn compute_hash(&self) -> Hash {
        let json = serde_json::to_string(self).expect("Failed to serialize commit");
        Sha256::digest(json.as_bytes()).to_vec()
    }
}

/// Protocol for storing commits and references
pub trait CommitGraphStore: Send + Sync {
    /// Store a commit by its hash
    fn put_commit(&self, commit_hash: &Hash, commit: Commit);

    /// Retrieve a commit by its hash. Returns None if not found.
    fn get_commit(&self, commit_hash: &Hash) -> Option<Commit>;

    /// Get parent commit hashes for a given commit
    fn get_parents(&self, commit_hash: &Hash) -> Vec<Hash>;

    /// Set a reference (branch/tag) to point to a commit
    fn set_ref(&self, name: &str, commit_hash: &Hash);

    /// Get the commit hash for a reference. Returns None if not found.
    fn get_ref(&self, name: &str) -> Option<Hash>;

    /// List all references and their commit hashes
    fn list_refs(&self) -> HashMap<String, Hash>;

    /// Set HEAD to point to a branch name
    fn set_head(&self, ref_name: &str);

    /// Get the branch name that HEAD points to. Returns None if not set.
    fn get_head(&self) -> Option<String>;

    /// Find a commit by its hash prefix (partial hash).
    ///
    /// # Arguments
    ///
    /// * `prefix` - Hex string prefix of the commit hash
    ///
    /// # Returns
    ///
    /// Full commit hash if exactly one match is found, None otherwise
    fn find_commit_by_prefix(&self, prefix: &str) -> Option<Hash>;
}

/// In-memory implementation of CommitGraphStore
#[derive(Clone)]
pub struct MemoryCommitGraphStore {
    commits: Arc<Mutex<HashMap<Hash, Commit>>>,
    refs: Arc<Mutex<HashMap<String, Hash>>>,
    head: Arc<Mutex<Option<String>>>,
}

impl MemoryCommitGraphStore {
    /// Create a new in-memory commit graph store
    pub fn new() -> Self {
        MemoryCommitGraphStore {
            commits: Arc::new(Mutex::new(HashMap::new())),
            refs: Arc::new(Mutex::new(HashMap::new())),
            head: Arc::new(Mutex::new(None)),
        }
    }
}

impl Default for MemoryCommitGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CommitGraphStore for MemoryCommitGraphStore {
    fn put_commit(&self, commit_hash: &Hash, commit: Commit) {
        let mut commits = self.commits.lock().unwrap();
        commits.insert(commit_hash.clone(), commit);
    }

    fn get_commit(&self, commit_hash: &Hash) -> Option<Commit> {
        let commits = self.commits.lock().unwrap();
        commits.get(commit_hash).cloned()
    }

    fn get_parents(&self, commit_hash: &Hash) -> Vec<Hash> {
        self.get_commit(commit_hash)
            .map(|c| c.parents)
            .unwrap_or_default()
    }

    fn set_ref(&self, name: &str, commit_hash: &Hash) {
        let mut refs = self.refs.lock().unwrap();
        refs.insert(name.to_string(), commit_hash.clone());
    }

    fn get_ref(&self, name: &str) -> Option<Hash> {
        let refs = self.refs.lock().unwrap();
        refs.get(name).cloned()
    }

    fn list_refs(&self) -> HashMap<String, Hash> {
        let refs = self.refs.lock().unwrap();
        refs.clone()
    }

    fn set_head(&self, ref_name: &str) {
        let mut head = self.head.lock().unwrap();
        *head = Some(ref_name.to_string());
    }

    fn get_head(&self) -> Option<String> {
        let head = self.head.lock().unwrap();
        head.clone()
    }

    fn find_commit_by_prefix(&self, prefix: &str) -> Option<Hash> {
        let commits = self.commits.lock().unwrap();
        let matches: Vec<_> = commits
            .keys()
            .filter(|hash| hex::encode(hash).starts_with(prefix))
            .collect();

        if matches.len() == 1 {
            Some(matches[0].clone())
        } else {
            None
        }
    }
}

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
    pub fn new<P: AsRef<Path>>(db_path: P) -> crate::Result<Self> {
        let conn = Connection::open(db_path)?;
        let store = SqliteCommitGraphStore {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.create_tables()?;
        Ok(store)
    }

    fn create_tables(&self) -> crate::Result<()> {
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
        conn.execute("DELETE FROM commit_parents WHERE commit_hash = ?1", params![commit_hash])
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
            .prepare("SELECT tree_root, message, timestamp, author, pattern, seed FROM commits WHERE hash = ?1")
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
            .prepare("SELECT parent_hash FROM commit_parents WHERE commit_hash = ?1 ORDER BY parent_index")
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
            .prepare("SELECT parent_hash FROM commit_parents WHERE commit_hash = ?1 ORDER BY parent_index")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_hash() {
        let commit = Commit::new(
            vec![1, 2, 3],
            vec![],
            "Test commit".to_string(),
            1234567890.0,
            "test@example.com".to_string(),
        );

        let hash1 = commit.compute_hash();
        let hash2 = commit.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_memory_store() {
        let store = MemoryCommitGraphStore::new();

        let commit = Commit::new(
            vec![1, 2, 3],
            vec![],
            "Test commit".to_string(),
            1234567890.0,
            "test@example.com".to_string(),
        );

        let hash = commit.compute_hash();
        store.put_commit(&hash, commit.clone());

        let retrieved = store.get_commit(&hash);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().message, "Test commit");
    }

    #[test]
    fn test_refs() {
        let store = MemoryCommitGraphStore::new();

        let commit_hash = vec![1, 2, 3, 4];
        store.set_ref("main", &commit_hash);

        let retrieved = store.get_ref("main");
        assert_eq!(retrieved, Some(commit_hash.clone()));

        let refs = store.list_refs();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs.get("main"), Some(&commit_hash));
    }

    #[test]
    fn test_head() {
        let store = MemoryCommitGraphStore::new();

        assert_eq!(store.get_head(), None);

        store.set_head("main");
        assert_eq!(store.get_head(), Some("main".to_string()));
    }

    #[test]
    fn test_find_by_prefix() {
        let store = MemoryCommitGraphStore::new();

        let commit = Commit::new(
            vec![1, 2, 3],
            vec![],
            "Test".to_string(),
            1234567890.0,
            "test".to_string(),
        );

        let hash = commit.compute_hash();
        store.put_commit(&hash, commit);

        let hex_hash = hex::encode(&hash);
        let prefix = &hex_hash[..8];

        let found = store.find_commit_by_prefix(prefix);
        assert_eq!(found, Some(hash));
    }
}
