//! Garbage collection for ProllyTree.
//!
//! Identifies unreachable nodes in the store by traversing from root hashes
//! and comparing with all stored nodes.

use std::collections::HashSet;

use crate::store::BlockStore;
use crate::Hash;

/// Statistics from a garbage collection operation
#[derive(Debug, Clone)]
pub struct GCStats {
    pub total_nodes: usize,
    pub reachable_nodes: usize,
    pub garbage_nodes: usize,
}

impl GCStats {
    /// Percentage of nodes that are reachable
    pub fn reachable_percent(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            (self.reachable_nodes as f64 / self.total_nodes as f64) * 100.0
        }
    }

    /// Percentage of nodes that are garbage
    pub fn garbage_percent(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            (self.garbage_nodes as f64 / self.total_nodes as f64) * 100.0
        }
    }
}

/// Find all nodes reachable from a set of root hashes.
///
/// Performs a depth-first traversal from each root, visiting all
/// descendant nodes and collecting their hashes.
///
/// # Arguments
///
/// * `store` - Storage backend containing nodes
/// * `root_hashes` - Set of root hashes to start traversal from
///
/// # Returns
///
/// Set of all reachable node hashes (including roots)
pub fn find_reachable_nodes(store: &dyn BlockStore, root_hashes: &HashSet<Hash>) -> HashSet<Hash> {
    let mut reachable: HashSet<Hash> = HashSet::new();
    let mut to_visit: Vec<Hash> = root_hashes.iter().cloned().collect();

    while let Some(node_hash) = to_visit.pop() {
        // Skip if already visited
        if reachable.contains(&node_hash) {
            continue;
        }

        // Get node and traverse children
        let node = match store.get_node(&node_hash) {
            Some(n) => n,
            None => continue, // Node not found - skip it
        };

        // Mark as reachable (only if node exists)
        reachable.insert(node_hash);

        if !node.is_leaf {
            // Internal node - add all children to visit
            for child_hash in &node.values {
                let hash_vec = child_hash.to_vec();
                if !reachable.contains(&hash_vec) {
                    to_visit.push(hash_vec);
                }
            }
        }
    }

    reachable
}

/// Find all garbage (unreachable) nodes in the store.
///
/// # Arguments
///
/// * `store` - Storage backend containing nodes
/// * `root_hashes` - Set of root hashes to keep (reachable roots)
///
/// # Returns
///
/// Set of garbage node hashes (unreachable from any root)
pub fn find_garbage_nodes(store: &dyn BlockStore, root_hashes: &HashSet<Hash>) -> HashSet<Hash> {
    // Find all reachable nodes
    let reachable = find_reachable_nodes(store, root_hashes);

    // Find all nodes in store
    let all_nodes: HashSet<Hash> = store.list_nodes().into_iter().collect();

    // Garbage = all nodes - reachable nodes
    all_nodes.difference(&reachable).cloned().collect()
}

/// Compute garbage collection statistics without removing anything.
///
/// # Arguments
///
/// * `store` - Storage backend containing nodes
/// * `root_hashes` - Set of root hashes to keep
///
/// # Returns
///
/// GCStats with information about reachable and garbage nodes
pub fn collect_garbage_stats(store: &dyn BlockStore, root_hashes: &HashSet<Hash>) -> GCStats {
    // Find reachable and all nodes
    let reachable = find_reachable_nodes(store, root_hashes);
    let all_nodes: HashSet<Hash> = store.list_nodes().into_iter().collect();

    // Compute statistics
    GCStats {
        total_nodes: all_nodes.len(),
        reachable_nodes: reachable.len(),
        garbage_nodes: all_nodes.len() - reachable.len(),
    }
}

/// Remove garbage nodes from the store.
///
/// Note: This is a destructive operation and should only be called
/// after verifying that the nodes are truly garbage.
///
/// # Arguments
///
/// * `store` - Storage backend to remove nodes from
/// * `garbage_hashes` - Set of node hashes to remove
///
/// # Returns
///
/// Number of nodes actually removed
pub fn remove_garbage(store: &dyn BlockStore, garbage_hashes: &HashSet<Hash>) -> usize {
    let mut removed_count = 0;

    for node_hash in garbage_hashes {
        if store.delete_node(node_hash) {
            removed_count += 1;
        }
    }

    removed_count
}

/// Perform garbage collection on a store.
///
/// # Arguments
///
/// * `store` - Storage backend to garbage collect
/// * `root_hashes` - Set of root hashes to keep (everything else is garbage)
/// * `dry_run` - If true, only compute statistics without removing nodes
///
/// # Returns
///
/// GCStats with information about the garbage collection operation
pub fn garbage_collect(
    store: &dyn BlockStore,
    root_hashes: &HashSet<Hash>,
    dry_run: bool,
) -> GCStats {
    // Compute statistics
    let stats = collect_garbage_stats(store, root_hashes);

    // If not dry run, actually remove garbage
    if !dry_run {
        let garbage = find_garbage_nodes(store, root_hashes);
        let removed_count = remove_garbage(store, &garbage);

        // Verify removal count matches expectations
        if removed_count != stats.garbage_nodes {
            panic!(
                "Expected to remove {} nodes, but actually removed {}",
                stats.garbage_nodes, removed_count
            );
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use crate::ProllyTree;
    use std::sync::Arc;

    #[test]
    fn test_find_reachable_empty() {
        let store = MemoryBlockStore::new();
        let roots = HashSet::new();
        let reachable = find_reachable_nodes(&store as &dyn BlockStore, &roots);
        assert_eq!(reachable.len(), 0);
    }

    #[test]
    fn test_gc_no_garbage() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut tree = ProllyTree::new(0.01, 42, Some(store.clone()));

        tree.insert_batch(
            vec![
                (b"a".to_vec(), b"val_a".to_vec()),
                (b"b".to_vec(), b"val_b".to_vec()),
            ],
            false,
        );

        let root_hash = tree.get_root_hash();
        let mut roots = HashSet::new();
        roots.insert(root_hash);

        let stats = collect_garbage_stats(store.as_ref(), &roots);

        assert_eq!(stats.garbage_nodes, 0);
        assert!(stats.reachable_nodes > 0);
        assert_eq!(stats.total_nodes, stats.reachable_nodes);
    }

    #[test]
    fn test_gc_with_garbage() {
        let store = Arc::new(MemoryBlockStore::new());

        // Create first tree
        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(vec![(b"a".to_vec(), b"val_a".to_vec())], false);
        let _old_root = tree1.get_root_hash();

        // Create second tree (with different data)
        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(
            vec![
                (b"x".to_vec(), b"val_x".to_vec()),
                (b"y".to_vec(), b"val_y".to_vec()),
            ],
            false,
        );
        let new_root = tree2.get_root_hash();

        // Only keep new root - old root's nodes are garbage
        let mut roots = HashSet::new();
        roots.insert(new_root);

        let stats = collect_garbage_stats(store.as_ref(), &roots);

        assert!(stats.garbage_nodes > 0, "Should have garbage nodes");
        assert!(stats.reachable_nodes > 0, "Should have reachable nodes");
        assert_eq!(
            stats.total_nodes,
            stats.reachable_nodes + stats.garbage_nodes
        );
    }

    #[test]
    fn test_gc_remove_garbage() {
        let store = Arc::new(MemoryBlockStore::new());

        // Create first tree
        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(vec![(b"a".to_vec(), b"val_a".to_vec())], false);

        // Create second tree
        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(vec![(b"x".to_vec(), b"val_x".to_vec())], false);
        let new_root = tree2.get_root_hash();

        let initial_count = store.count_nodes();

        // GC, keeping only new root
        let mut roots = HashSet::new();
        roots.insert(new_root);

        let stats = garbage_collect(store.as_ref(), &roots, false);

        let final_count = store.count_nodes();

        assert_eq!(final_count, stats.reachable_nodes);
        assert_eq!(initial_count - final_count, stats.garbage_nodes);
    }

    #[test]
    fn test_gc_dry_run() {
        let store = Arc::new(MemoryBlockStore::new());

        // Create two trees
        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(vec![(b"a".to_vec(), b"val_a".to_vec())], false);

        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(vec![(b"x".to_vec(), b"val_x".to_vec())], false);
        let new_root = tree2.get_root_hash();

        let initial_count = store.count_nodes();

        // Dry run GC
        let mut roots = HashSet::new();
        roots.insert(new_root);

        let stats = garbage_collect(store.as_ref(), &roots, true);

        let final_count = store.count_nodes();

        // Dry run should not remove anything
        assert_eq!(initial_count, final_count);
        assert!(stats.garbage_nodes > 0);
    }

    #[test]
    fn test_gc_with_repo() {
        use crate::{MemoryCommitGraphStore, Repo};

        let block_store = Arc::new(MemoryBlockStore::new());
        let commit_store = Arc::new(MemoryCommitGraphStore::new());
        let repo = Repo::init_empty(
            block_store.clone(),
            commit_store,
            "test@example.com".to_string(),
        );

        // Create several commits on main
        for i in 0..3 {
            let mut tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
            tree.insert_batch(
                vec![(format!("key{}", i).into_bytes(), b"value".to_vec())],
                false,
            );
            let new_root = tree.get_root_hash();
            repo.commit(&new_root, &format!("Commit {}", i), None, None, None);
        }

        // Create a branch from an earlier commit
        repo.create_branch("old", Some(&repo.resolve_ref("HEAD~2").unwrap()))
            .unwrap();

        // GC should keep all nodes reachable from any ref (main and old)
        let tree_roots = repo.get_reachable_tree_roots();
        let stats = garbage_collect(block_store.as_ref(), &tree_roots, false);

        // Should have no garbage since all commits are reachable from refs
        assert_eq!(stats.garbage_nodes, 0);

        // Now create an orphaned tree that's not part of any commit
        let mut orphan_tree = ProllyTree::new(0.01, 42, Some(block_store.clone()));
        orphan_tree.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);
        let _orphan_root = orphan_tree.get_root_hash();

        let nodes_with_orphan = block_store.count_nodes();

        // GC should remove the orphan nodes
        let tree_roots = repo.get_reachable_tree_roots();
        let stats = garbage_collect(block_store.as_ref(), &tree_roots, false);

        let final_count = block_store.count_nodes();

        // Verify we removed the orphan nodes
        assert!(stats.garbage_nodes > 0, "Should have removed orphan nodes");
        assert_eq!(
            nodes_with_orphan - final_count,
            stats.garbage_nodes,
            "Should have removed exactly the garbage nodes"
        );

        // All remaining nodes should be reachable from refs
        let reachable = find_reachable_nodes(block_store.as_ref(), &tree_roots);
        assert_eq!(reachable.len(), final_count);
    }
}
