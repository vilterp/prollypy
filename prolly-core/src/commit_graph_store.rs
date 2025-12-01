//! Commit graph store implementations for ProllyPy.
//!
//! Provides storage backends for commits and references in the version control system.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::StoreResult;
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
        // This uses serde_json which shouldn't fail for this struct,
        // but if it does, we'll produce a predictable hash based on fields
        match serde_json::to_string(self) {
            Ok(json) => Sha256::digest(json.as_bytes()).to_vec(),
            Err(_) => {
                // Fallback: hash the critical fields directly
                let mut hasher = Sha256::new();
                hasher.update(&self.tree_root);
                hasher.update(self.message.as_bytes());
                hasher.update(self.author.as_bytes());
                hasher.update(self.timestamp.to_le_bytes());
                hasher.finalize().to_vec()
            }
        }
    }
}

/// Protocol for storing commits and references
pub trait CommitGraphStore: Send + Sync {
    /// Store a commit by its hash
    fn put_commit(&self, commit_hash: &Hash, commit: Commit) -> StoreResult<()>;

    /// Retrieve a commit by its hash. Returns None if not found.
    fn get_commit(&self, commit_hash: &Hash) -> StoreResult<Option<Commit>>;

    /// Get parent commit hashes for a given commit
    fn get_parents(&self, commit_hash: &Hash) -> StoreResult<Vec<Hash>>;

    /// Set a reference (branch/tag) to point to a commit
    fn set_ref(&self, name: &str, commit_hash: &Hash) -> StoreResult<()>;

    /// Get the commit hash for a reference. Returns None if not found.
    fn get_ref(&self, name: &str) -> StoreResult<Option<Hash>>;

    /// List all references and their commit hashes
    fn list_refs(&self) -> StoreResult<HashMap<String, Hash>>;

    /// Set HEAD to point to a branch name
    fn set_head(&self, ref_name: &str) -> StoreResult<()>;

    /// Get the branch name that HEAD points to. Returns None if not set.
    fn get_head(&self) -> StoreResult<Option<String>>;

    /// Find a commit by its hash prefix (partial hash).
    ///
    /// # Arguments
    ///
    /// * `prefix` - Hex string prefix of the commit hash
    ///
    /// # Returns
    ///
    /// Full commit hash if exactly one match is found, None otherwise
    fn find_commit_by_prefix(&self, prefix: &str) -> StoreResult<Option<Hash>>;
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
    fn put_commit(&self, commit_hash: &Hash, commit: Commit) -> StoreResult<()> {
        let mut commits = self.commits.lock()?;
        commits.insert(commit_hash.clone(), commit);
        Ok(())
    }

    fn get_commit(&self, commit_hash: &Hash) -> StoreResult<Option<Commit>> {
        let commits = self.commits.lock()?;
        Ok(commits.get(commit_hash).cloned())
    }

    fn get_parents(&self, commit_hash: &Hash) -> StoreResult<Vec<Hash>> {
        Ok(self.get_commit(commit_hash)?
            .map(|c| c.parents)
            .unwrap_or_default())
    }

    fn set_ref(&self, name: &str, commit_hash: &Hash) -> StoreResult<()> {
        let mut refs = self.refs.lock()?;
        refs.insert(name.to_string(), commit_hash.clone());
        Ok(())
    }

    fn get_ref(&self, name: &str) -> StoreResult<Option<Hash>> {
        let refs = self.refs.lock()?;
        Ok(refs.get(name).cloned())
    }

    fn list_refs(&self) -> StoreResult<HashMap<String, Hash>> {
        let refs = self.refs.lock()?;
        Ok(refs.clone())
    }

    fn set_head(&self, ref_name: &str) -> StoreResult<()> {
        let mut head = self.head.lock()?;
        *head = Some(ref_name.to_string());
        Ok(())
    }

    fn get_head(&self) -> StoreResult<Option<String>> {
        let head = self.head.lock()?;
        Ok(head.clone())
    }

    fn find_commit_by_prefix(&self, prefix: &str) -> StoreResult<Option<Hash>> {
        let commits = self.commits.lock()?;
        let matches: Vec<_> = commits
            .keys()
            .filter(|hash| hex::encode(hash).starts_with(prefix))
            .collect();

        if matches.len() == 1 {
            Ok(Some(matches[0].clone()))
        } else {
            Ok(None)
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
        store.put_commit(&hash, commit.clone()).unwrap();

        let retrieved = store.get_commit(&hash).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().message, "Test commit");
    }

    #[test]
    fn test_refs() {
        let store = MemoryCommitGraphStore::new();

        let commit_hash = vec![1, 2, 3, 4];
        store.set_ref("main", &commit_hash).unwrap();

        let retrieved = store.get_ref("main").unwrap();
        assert_eq!(retrieved, Some(commit_hash.clone()));

        let refs = store.list_refs().unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs.get("main"), Some(&commit_hash));
    }

    #[test]
    fn test_head() {
        let store = MemoryCommitGraphStore::new();

        assert_eq!(store.get_head().unwrap(), None);

        store.set_head("main").unwrap();
        assert_eq!(store.get_head().unwrap(), Some("main".to_string()));
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
        store.put_commit(&hash, commit).unwrap();

        let hex_hash = hex::encode(&hash);
        let prefix = &hex_hash[..8];

        let found = store.find_commit_by_prefix(prefix).unwrap();
        assert_eq!(found, Some(hash));
    }
}
