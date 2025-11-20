//! Repository abstraction for version-controlled ProllyTrees.
//!
//! Provides git-like version control capabilities on top of ProllyTree.
//! Includes commit tracking, branching, and reference management.

use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::commit_graph_store::{Commit, CommitGraphStore};
use crate::store::BlockStore;
use crate::tree::ProllyTree;
use crate::Hash;

/// Repository for version-controlled ProllyTrees.
///
/// Provides git-like operations: commit, branch, and reference management.
pub struct Repo {
    pub block_store: Arc<dyn BlockStore>,
    pub commit_graph_store: Arc<dyn CommitGraphStore>,
    pub default_author: String,
}

impl Repo {
    /// Initialize repository.
    ///
    /// # Arguments
    ///
    /// * `block_store` - Storage for tree nodes
    /// * `commit_graph_store` - Storage for commits and refs
    /// * `default_author` - Default author for commits
    pub fn new(
        block_store: Arc<dyn BlockStore>,
        commit_graph_store: Arc<dyn CommitGraphStore>,
        default_author: String,
    ) -> Self {
        Repo {
            block_store,
            commit_graph_store,
            default_author,
        }
    }

    /// Initialize a new empty repository with an initial commit.
    ///
    /// # Arguments
    ///
    /// * `block_store` - Storage for tree nodes
    /// * `commit_graph_store` - Storage for commits and refs
    /// * `default_author` - Default author for commits
    pub fn init_empty(
        block_store: Arc<dyn BlockStore>,
        commit_graph_store: Arc<dyn CommitGraphStore>,
        default_author: String,
    ) -> Self {
        let repo = Self::new(block_store.clone(), commit_graph_store.clone(), default_author.clone());

        // Create empty tree
        let empty_tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
        let tree_root = empty_tree.get_root_hash();

        // Create initial commit
        let initial_commit = Commit {
            tree_root: tree_root.clone(),
            parents: vec![],
            message: "Initial commit".to_string(),
            timestamp: current_timestamp(),
            author: default_author,
            pattern: 0.01,
            seed: 42,
        };

        let commit_hash = initial_commit.compute_hash();
        commit_graph_store.put_commit(&commit_hash, initial_commit);

        // Create main branch and set HEAD
        commit_graph_store.set_ref("main", &commit_hash);
        commit_graph_store.set_head("main");

        repo
    }

    /// Get the current HEAD commit and ref name.
    ///
    /// # Returns
    ///
    /// Tuple of (commit, ref_name). Commit may be None if ref doesn't exist yet.
    pub fn get_head(&self) -> (Option<Commit>, String) {
        let current_ref = self.commit_graph_store.get_head()
            .unwrap_or_else(|| {
                // HEAD not set, default to "main" and set it
                let default_ref = "main".to_string();
                self.commit_graph_store.set_head(&default_ref);
                default_ref
            });

        let commit = self.commit_graph_store.get_ref(&current_ref)
            .and_then(|hash| self.commit_graph_store.get_commit(&hash));

        (commit, current_ref)
    }

    /// Create a new branch.
    ///
    /// # Arguments
    ///
    /// * `name` - Branch name
    /// * `from_commit` - Commit hash to branch from. If None, uses current HEAD.
    ///
    /// # Errors
    ///
    /// Returns error if branch already exists or from_commit is invalid
    pub fn create_branch(&self, name: &str, from_commit: Option<&Hash>) -> crate::Result<()> {
        // Check if branch already exists
        if self.commit_graph_store.get_ref(name).is_some() {
            return Err(format!("Branch '{}' already exists", name).into());
        }

        // Determine commit to branch from
        let commit_hash = if let Some(hash) = from_commit {
            hash.clone()
        } else {
            let (head_commit, _) = self.get_head();
            if let Some(commit) = head_commit {
                commit.compute_hash()
            } else {
                return Err("Cannot create branch: no commits exist yet".into());
            }
        };

        // Verify the commit exists
        if self.commit_graph_store.get_commit(&commit_hash).is_none() {
            return Err(format!("Commit {} does not exist", hex::encode(&commit_hash)).into());
        }

        // Create the branch
        self.commit_graph_store.set_ref(name, &commit_hash);
        Ok(())
    }

    /// Switch to a different branch.
    ///
    /// # Arguments
    ///
    /// * `ref_name` - Name of the branch to checkout
    ///
    /// # Errors
    ///
    /// Returns error if ref doesn't exist
    pub fn checkout(&self, ref_name: &str) -> crate::Result<()> {
        if self.commit_graph_store.get_ref(ref_name).is_none() {
            return Err(format!("Ref '{}' does not exist", ref_name).into());
        }

        self.commit_graph_store.set_head(ref_name);
        Ok(())
    }

    /// Create a new commit with the given tree root.
    ///
    /// # Arguments
    ///
    /// * `new_head_tree` - Hash of the prolly tree root for this commit
    /// * `message` - Commit message
    /// * `author` - Author name/email. Uses default if not provided.
    /// * `pattern` - Split probability for the prolly tree. If None, inherits from parent.
    /// * `seed` - Seed for rolling hash function. If None, inherits from parent.
    ///
    /// # Returns
    ///
    /// The newly created Commit object
    pub fn commit(
        &self,
        new_head_tree: &Hash,
        message: &str,
        author: Option<&str>,
        pattern: Option<f64>,
        seed: Option<u32>,
    ) -> Commit {
        // Get current HEAD as parent
        let (head_commit, ref_name) = self.get_head();
        let parents = if let Some(commit) = &head_commit {
            vec![commit.compute_hash()]
        } else {
            vec![]
        };

        // Inherit pattern and seed from parent if not specified
        let pattern = pattern.unwrap_or_else(|| {
            head_commit.as_ref().map(|c| c.pattern).unwrap_or(0.01)
        });
        let seed = seed.unwrap_or_else(|| {
            head_commit.as_ref().map(|c| c.seed).unwrap_or(42)
        });

        // Create new commit
        let new_commit = Commit {
            tree_root: new_head_tree.clone(),
            parents,
            message: message.to_string(),
            timestamp: current_timestamp(),
            author: author.unwrap_or(&self.default_author).to_string(),
            pattern,
            seed,
        };

        // Compute commit hash and store it
        let commit_hash = new_commit.compute_hash();
        self.commit_graph_store.put_commit(&commit_hash, new_commit.clone());

        // Update current ref to point to new commit
        self.commit_graph_store.set_ref(&ref_name, &commit_hash);

        new_commit
    }

    /// Get a commit by its hash.
    pub fn get_commit(&self, commit_hash: &Hash) -> Option<Commit> {
        self.commit_graph_store.get_commit(commit_hash)
    }

    /// List all branches and their commit hashes.
    pub fn list_branches(&self) -> Vec<(String, Hash)> {
        self.commit_graph_store.list_refs()
            .into_iter()
            .collect()
    }

    /// Resolve a ref name or commit hash to a commit hash.
    ///
    /// Supports:
    /// - Branch/tag names (e.g., "main", "develop")
    /// - "HEAD" - resolves to the commit of the current HEAD ref
    /// - "HEAD~" or "HEAD~N" - resolves to the Nth parent of HEAD (default N=1)
    /// - Full commit hashes (64 hex characters)
    /// - Partial commit hashes (e.g., "341e719a") - must match exactly one commit
    ///
    /// # Arguments
    ///
    /// * `ref_str` - Branch name, "HEAD", or commit hash (full or partial hex string)
    ///
    /// # Returns
    ///
    /// Commit hash, or None if not found
    pub fn resolve_ref(&self, ref_str: &str) -> Option<Hash> {
        // Handle HEAD~ syntax (HEAD~, HEAD~1, HEAD~2, etc.)
        if ref_str.starts_with("HEAD~") {
            // Parse the number of parents to go back
            let n = if ref_str == "HEAD~" {
                1
            } else {
                ref_str[5..].parse::<usize>().ok()?
            };

            // Start with HEAD
            let (head_commit, _) = self.get_head();
            let mut current_commit_hash = head_commit?.compute_hash();

            // Walk back n parents
            for _ in 0..n {
                let current_commit = self.get_commit(&current_commit_hash)?;
                if current_commit.parents.is_empty() {
                    return None; // No parent available
                }
                // Take the first parent
                current_commit_hash = current_commit.parents[0].clone();
            }

            return Some(current_commit_hash);
        }

        // Handle HEAD specially
        if ref_str == "HEAD" {
            let (head_commit, _) = self.get_head();
            return head_commit.map(|c| c.compute_hash());
        }

        // Try as a ref (branch/tag) first
        if let Some(commit_hash) = self.commit_graph_store.get_ref(ref_str) {
            return Some(commit_hash);
        }

        // Try as a full commit hash
        if let Ok(commit_hash) = hex::decode(ref_str) {
            if self.commit_graph_store.get_commit(&commit_hash).is_some() {
                return Some(commit_hash);
            }
        }

        // Try as a partial commit hash
        if ref_str.chars().all(|c| c.is_ascii_hexdigit()) {
            return self.commit_graph_store.find_commit_by_prefix(ref_str);
        }

        None
    }

    /// Walk the commit graph starting from a frontier set, yielding commits
    /// in reverse chronological order (most recent first).
    ///
    /// This method uses a priority queue to ensure commits are yielded in
    /// timestamp order (newest first), which is important for garbage collection
    /// and other operations that need to traverse the commit graph in a specific order.
    ///
    /// # Arguments
    ///
    /// * `frontier` - Set of commit hashes to start walking from
    ///
    /// # Returns
    ///
    /// Vector of (commit_hash, commit) tuples in reverse chronological order
    pub fn walk(&self, frontier: &HashSet<Hash>) -> Vec<(Hash, Commit)> {
        // Priority queue: (Reverse(timestamp), commit_hash)
        // We use Reverse to get max-heap behavior (most recent first)
        let mut heap: BinaryHeap<CommitEntry> = BinaryHeap::new();
        let mut visited: HashSet<Hash> = HashSet::new();
        let mut result = Vec::new();

        // Initialize heap with frontier commits
        for commit_hash in frontier {
            if let Some(commit) = self.commit_graph_store.get_commit(commit_hash) {
                heap.push(CommitEntry {
                    timestamp: commit.timestamp,
                    hash: commit_hash.clone(),
                });
            }
        }

        // Process commits in timestamp order (most recent first)
        while let Some(entry) = heap.pop() {
            let commit_hash = entry.hash;

            // Skip if already visited
            if visited.contains(&commit_hash) {
                continue;
            }

            visited.insert(commit_hash.clone());

            // Get the commit
            let commit = match self.commit_graph_store.get_commit(&commit_hash) {
                Some(c) => c,
                None => continue,
            };

            result.push((commit_hash.clone(), commit.clone()));

            // Add parents to the heap
            for parent_hash in &commit.parents {
                if !visited.contains(parent_hash) {
                    if let Some(parent_commit) = self.commit_graph_store.get_commit(parent_hash) {
                        heap.push(CommitEntry {
                            timestamp: parent_commit.timestamp,
                            hash: parent_hash.clone(),
                        });
                    }
                }
            }
        }

        result
    }

    /// Walk the commit graph backwards from a starting point, yielding commits.
    ///
    /// This is implemented in terms of walk() by starting with a frontier of {HEAD}
    /// or the specified ref.
    ///
    /// # Arguments
    ///
    /// * `start_ref` - Ref or commit hash to start from. If None, uses HEAD.
    /// * `max_count` - Maximum number of commits to return. If None, returns all.
    ///
    /// # Returns
    ///
    /// Vector of (commit_hash, commit) tuples in reverse chronological order
    pub fn log(&self, start_ref: Option<&str>, max_count: Option<usize>) -> Vec<(Hash, Commit)> {
        // Determine starting commit
        let current_hash = if let Some(ref_str) = start_ref {
            match self.resolve_ref(ref_str) {
                Some(hash) => hash,
                None => return vec![],
            }
        } else {
            let (head_commit, _) = self.get_head();
            match head_commit {
                Some(commit) => commit.compute_hash(),
                None => return vec![],
            }
        };

        // Create frontier with just the starting commit
        let mut frontier = HashSet::new();
        frontier.insert(current_hash);

        // Walk and return commits, respecting max_count
        let commits = self.walk(&frontier);
        if let Some(max) = max_count {
            commits.into_iter().take(max).collect()
        } else {
            commits
        }
    }

    /// Get all tree roots that are reachable from any ref.
    ///
    /// This walks the commit graph starting from all refs and collects
    /// all tree roots. This is useful for garbage collection to identify
    /// which prolly tree nodes should be kept.
    ///
    /// # Returns
    ///
    /// Set of tree root hashes that are reachable
    pub fn get_reachable_tree_roots(&self) -> HashSet<Hash> {
        // Get all refs as the frontier
        let refs = self.commit_graph_store.list_refs();
        if refs.is_empty() {
            return HashSet::new();
        }

        let frontier: HashSet<Hash> = refs.into_iter().map(|(_, hash)| hash).collect();

        // Walk through all commits and collect tree roots
        let mut tree_roots: HashSet<Hash> = HashSet::new();
        for (_commit_hash, commit) in self.walk(&frontier) {
            tree_roots.insert(commit.tree_root);
        }

        tree_roots
    }
}

/// Helper struct for priority queue in walk()
#[derive(PartialEq)]
struct CommitEntry {
    timestamp: f64,
    hash: Hash,
}

impl Eq for CommitEntry {}

// Implement Ord for max-heap behavior (most recent timestamp first)
impl Ord for CommitEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by timestamp (higher = more recent = higher priority)
        // Use partial_cmp and unwrap because f64 doesn't implement Ord
        // (due to NaN), but our timestamps should always be valid
        self.timestamp
            .partial_cmp(&other.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for CommitEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Get current Unix timestamp as f64
fn current_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryBlockStore, MemoryCommitGraphStore};

    #[test]
    fn test_init_empty() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store.clone(), commit_store.clone(), "test@example.com".to_string());

        // Should have a main branch
        let refs = repo.list_branches();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].0, "main");

        // HEAD should point to main
        let (commit, ref_name) = repo.get_head();
        assert_eq!(ref_name, "main");
        assert!(commit.is_some());
        assert_eq!(commit.unwrap().message, "Initial commit");
    }

    #[test]
    fn test_create_branch() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store, commit_store, "test@example.com".to_string());

        // Create a new branch
        repo.create_branch("develop", None).unwrap();

        let refs = repo.list_branches();
        assert_eq!(refs.len(), 2);

        // Both should point to the same commit
        let main_hash = repo.resolve_ref("main").unwrap();
        let develop_hash = repo.resolve_ref("develop").unwrap();
        assert_eq!(main_hash, develop_hash);
    }

    #[test]
    fn test_checkout() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store, commit_store, "test@example.com".to_string());

        repo.create_branch("develop", None).unwrap();
        repo.checkout("develop").unwrap();

        let (_, ref_name) = repo.get_head();
        assert_eq!(ref_name, "develop");
    }

    #[test]
    fn test_commit() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store.clone(), commit_store, "test@example.com".to_string());

        // Create a new tree
        let mut tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
        tree.insert_batch(vec![(b"key".to_vec(), b"value".to_vec())], false);
        let new_root = tree.get_root_hash();

        // Create a commit
        let commit = repo.commit(&new_root, "Test commit", None, None, None);
        assert_eq!(commit.message, "Test commit");
        assert_eq!(commit.tree_root, new_root);

        // Should have one parent (the initial commit)
        assert_eq!(commit.parents.len(), 1);

        // main should now point to the new commit
        let main_hash = repo.resolve_ref("main").unwrap();
        assert_eq!(main_hash, commit.compute_hash());
    }

    #[test]
    fn test_resolve_ref() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store.clone(), commit_store, "test@example.com".to_string());

        // Resolve HEAD
        let head_hash = repo.resolve_ref("HEAD").unwrap();
        let main_hash = repo.resolve_ref("main").unwrap();
        assert_eq!(head_hash, main_hash);

        // Create a second commit
        let mut tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
        tree.insert_batch(vec![(b"key".to_vec(), b"value".to_vec())], false);
        let new_root = tree.get_root_hash();
        repo.commit(&new_root, "Second commit", None, None, None);

        // Resolve HEAD~
        let parent_hash = repo.resolve_ref("HEAD~").unwrap();
        assert_eq!(parent_hash, main_hash);

        // Resolve HEAD~1 (same as HEAD~)
        let parent_hash_1 = repo.resolve_ref("HEAD~1").unwrap();
        assert_eq!(parent_hash_1, main_hash);
    }

    #[test]
    fn test_log() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store.clone(), commit_store, "test@example.com".to_string());

        // Create a few commits
        for i in 0..3 {
            let mut tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
            tree.insert_batch(vec![(format!("key{}", i).into_bytes(), b"value".to_vec())], false);
            let new_root = tree.get_root_hash();
            repo.commit(&new_root, &format!("Commit {}", i), None, None, None);
        }

        // Get full log
        let commits = repo.log(None, None);
        assert_eq!(commits.len(), 4); // 3 new + 1 initial

        // Get limited log
        let commits = repo.log(None, Some(2));
        assert_eq!(commits.len(), 2);

        // Verify order (most recent first)
        assert_eq!(commits[0].1.message, "Commit 2");
        assert_eq!(commits[1].1.message, "Commit 1");
    }

    #[test]
    fn test_get_reachable_tree_roots() {
        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(block_store.clone(), commit_store, "test@example.com".to_string());

        // Get initial tree root
        let (initial_commit, _) = repo.get_head();
        let initial_root = initial_commit.unwrap().tree_root;

        // Create a few commits on main
        let mut tree_roots = vec![];
        for i in 0..3 {
            let mut tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
            tree.insert_batch(vec![(format!("key{}", i).into_bytes(), b"value".to_vec())], false);
            let new_root = tree.get_root_hash();
            tree_roots.push(new_root.clone());
            repo.commit(&new_root, &format!("Commit {}", i), None, None, None);
        }

        // Create a branch and add another commit
        repo.create_branch("develop", Some(&repo.resolve_ref("HEAD~1").unwrap())).unwrap();
        repo.checkout("develop").unwrap();
        let mut tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
        tree.insert_batch(vec![(b"branch_key".to_vec(), b"value".to_vec())], false);
        let branch_root = tree.get_root_hash();
        repo.commit(&branch_root, "Branch commit", None, None, None);

        // Get reachable tree roots
        let reachable = repo.get_reachable_tree_roots();

        // Should include initial commit + 3 main commits + 1 branch commit
        // Note: some may be deduplicated if they share the same tree root
        assert!(reachable.len() >= 4);
        assert!(reachable.contains(&initial_root));
        assert!(reachable.contains(&branch_root));
    }
}
