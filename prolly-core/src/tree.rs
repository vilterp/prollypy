//! ProllyTree implementation with content-based splitting.
//!
//! Features:
//! - Content-addressed nodes (hash of contents)
//! - Rolling hash-based splitting (using SHA256)
//! - Incremental batch insert with subtree reuse
//! - Pluggable storage backends via BlockStore trait

use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Instant;

use crate::cursor::TreeCursor;
use crate::node::Node;
use crate::store::{BlockStore, MemoryBlockStore};
use crate::Hash;

const MIN_NODE_SIZE: usize = 2;
const TARGET_FANOUT: usize = 32;
const MIN_FANOUT: usize = 8;

/// Statistics for a single batch operation
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub nodes_created: usize,
    pub leaves_created: usize,
    pub internals_created: usize,
    pub nodes_reused: usize,
    pub subtrees_reused: usize,
    pub nodes_read: usize,
}

impl BatchStats {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn reset(&mut self) {
        *self = Default::default();
    }
}

/// ProllyTree with content-based splitting
pub struct ProllyTree {
    /// Split probability threshold (0.0 to 1.0 converted to u32)
    pattern: u32,
    /// Seed for rolling hash function
    seed: u32,
    /// Storage backend
    store: Arc<dyn BlockStore>,
    /// Root node
    pub root: Node,
    /// Operation statistics
    stats: BatchStats,
}

impl ProllyTree {
    /// Initialize ProllyTree with content-based splitting.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Split probability (0.0 to 1.0). Higher = smaller nodes, wider trees.
    ///              Default 0.01 means ~100 entries per node on average.
    /// * `seed` - Seed for rolling hash function for reproducibility
    /// * `store` - Storage backend (defaults to MemoryBlockStore if None)
    pub fn new(
        pattern: f64,
        seed: u32,
        store: Option<Arc<dyn BlockStore>>,
    ) -> Self {
        let pattern_u32 = (pattern * (u32::MAX as f64)) as u32;
        let store = store.unwrap_or_else(|| Arc::new(MemoryBlockStore::new()));

        ProllyTree {
            pattern: pattern_u32,
            seed,
            store,
            root: Node::new_leaf(),
            stats: BatchStats::new(),
        }
    }

    /// Create a new ProllyTree with default settings
    pub fn default() -> Self {
        Self::new(0.01, 42, None)
    }

    /// Reset operation statistics for a new batch
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Deterministic rolling hash using SHA256.
    ///
    /// # Arguments
    ///
    /// * `current_hash` - Current hash value (use self.seed for initial)
    /// * `data` - Bytes to add to the hash
    ///
    /// # Returns
    ///
    /// Updated hash value (u32)
    #[inline]
    fn rolling_hash(&self, current_hash: u32, data: &[u8]) -> u32 {
        // Optimized: use single-shot digest with chained updates
        let mut hasher = Sha256::new();
        hasher.update(current_hash.to_be_bytes());
        hasher.update(data);
        let hash_bytes = hasher.finalize();
        u32::from_be_bytes([hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3]])
    }

    /// Compute content hash for a node.
    ///
    /// For leaf nodes: hash the key-value pairs
    /// For internal nodes: hash the (separator_key, child_hash) pairs
    fn hash_node(&self, node: &Node) -> Hash {
        let mut combined = Vec::new();

        if node.is_leaf {
            // Hash key-value pairs
            for (key, value) in node.keys.iter().zip(node.values.iter()) {
                combined.extend_from_slice(key);
                combined.push(b':');
                combined.extend_from_slice(value);
                combined.push(b'|');
            }
        } else {
            // Hash separator keys and child hashes
            for (i, child_hash) in node.values.iter().enumerate() {
                if i < node.keys.len() {
                    combined.extend_from_slice(&node.keys[i]);
                } else {
                    combined.push(b'_');
                }
                combined.push(b':');
                combined.extend_from_slice(child_hash);
                combined.push(b'|');
            }
        }

        let hash = Sha256::digest(&combined);
        hash[..16].to_vec() // Use first 16 bytes
    }

    /// Store node and return its content-based hash
    fn store_node(&mut self, node: Node) -> Hash {
        let node_hash = self.hash_node(&node);

        // Only store if not already present (deduplication)
        if self.store.get_node(&node_hash).is_none() {
            self.store.put_node(&node_hash, node.clone());
            self.stats.nodes_created += 1;
            if node.is_leaf {
                self.stats.leaves_created += 1;
            } else {
                self.stats.internals_created += 1;
            }
        } else {
            self.stats.nodes_reused += 1;
        }

        node_hash
    }

    /// Incrementally insert a batch of (key, value) pairs.
    ///
    /// # Arguments
    ///
    /// * `mutations` - Sorted list of (key, value) tuples
    /// * `verbose` - Print progress information
    ///
    /// # Returns
    ///
    /// BatchStats with operation statistics
    pub fn insert_batch(&mut self, mutations: Vec<(Vec<u8>, Vec<u8>)>, verbose: bool) -> BatchStats {
        let start_time = Instant::now();

        // Reset stats for this batch
        self.reset_stats();

        // Rebuild tree with mutations
        let new_root = self.rebuild_with_mutations(self.root.clone(), &mutations);

        // Store the new root
        if !std::ptr::eq(&new_root as *const Node, &self.root as *const Node) {
            self.store_node(new_root.clone());
        }

        self.root = new_root;

        // Calculate timing
        let elapsed = start_time.elapsed();
        let rows_per_sec = if elapsed.as_secs_f64() > 0.0 {
            mutations.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        if verbose {
            println!(
                "Inserted {} rows; {} new nodes created; {:.0} rows/sec",
                mutations.len(),
                self.stats.nodes_created,
                rows_per_sec
            );
        }

        self.stats.clone()
    }

    /// Core incremental rebuild logic.
    ///
    /// # Returns
    ///
    /// New node (possibly with different structure)
    fn rebuild_with_mutations(&mut self, node: Node, mutations: &[(Vec<u8>, Vec<u8>)]) -> Node {
        if mutations.is_empty() {
            // No mutations for this subtree - REUSE it!
            return node;
        }

        if node.is_leaf {
            // Leaf node: merge old data with mutations
            let old_items: Vec<_> = node
                .keys
                .iter()
                .zip(node.values.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let merged = self.merge_sorted(&old_items, mutations);

            // Build new leaf nodes (may split if too large)
            let new_leaves = self.build_leaves(&merged);

            if new_leaves.len() == 1 {
                return new_leaves[0].clone();
            } else {
                // Multiple leaves - need parent (may recursively split if too many)
                return self.build_internal_from_children(new_leaves);
            }
        } else {
            // Internal node: partition mutations to children and rebuild recursively
            let mut new_children = Vec::new();
            let mut mut_idx = 0;

            for (i, child_hash) in node.values.iter().enumerate() {
                // Find mutations for this child
                let mut child_mutations = Vec::new();

                while mut_idx < mutations.len() {
                    let key = &mutations[mut_idx].0;

                    // Determine if this mutation belongs to this child
                    let belongs_here = if i == 0 {
                        // First child: all keys < first separator
                        node.keys.is_empty() || key.as_slice() < node.keys[0].as_slice()
                    } else if i < node.keys.len() {
                        // Middle child: separator[i-1] <= key < separator[i]
                        key.as_slice() < node.keys[i].as_slice()
                    } else {
                        // Last child: all remaining keys
                        true
                    };

                    if belongs_here {
                        child_mutations.push(mutations[mut_idx].clone());
                        mut_idx += 1;
                    } else {
                        break;
                    }
                }

                // Rebuild child (or reuse if no mutations)
                if let Some(child_node) = self.store.get_node(child_hash) {
                    let new_child = self.rebuild_with_mutations(child_node, &child_mutations);
                    new_children.push(new_child);
                }
            }

            // Now rebuild internal structure from new children
            if new_children.len() == 1 {
                // Single child - unwrap it
                return new_children[0].clone();
            } else {
                // Build new internal node(s) from children
                return self.build_internal_from_children(new_children);
            }
        }
    }

    /// Get the first actual key in a node's subtree.
    fn get_first_key(&self, node: &Node) -> Option<Vec<u8>> {
        if node.is_leaf {
            node.keys.first().cloned()
        } else {
            // Internal node - descend to leftmost child
            if node.values.is_empty() {
                return None;
            }
            let child_hash = &node.values[0];
            let child = self.store.get_node(child_hash)?;
            self.get_first_key(&child)
        }
    }

    /// Build a balanced internal tree from a list of children.
    fn build_internal_from_children(&mut self, children: Vec<Node>) -> Node {
        if children.is_empty() {
            panic!("Cannot build internal node with no children");
        }

        if children.len() == 1 {
            return children[0].clone();
        }

        // Calculate how many internal nodes we need at this level
        let num_internal_nodes = std::cmp::max(2, (children.len() + TARGET_FANOUT - 1) / TARGET_FANOUT);

        // Calculate children per internal node (distribute evenly)
        let children_per_node = children.len() / num_internal_nodes;
        let extra_children = children.len() % num_internal_nodes;

        // Build internal nodes with balanced distribution
        let mut internal_nodes = Vec::new();
        let mut child_idx = 0;

        for node_num in 0..num_internal_nodes {
            // Some nodes get one extra child to distribute the remainder
            let node_size = children_per_node + if node_num < extra_children { 1 } else { 0 };

            // Don't create nodes that are too small (merge with previous)
            if node_size < MIN_FANOUT && !internal_nodes.is_empty() && node_num == num_internal_nodes - 1 {
                // Last node is too small - merge with previous node
                let node_children = &children[child_idx..child_idx + node_size];
                let mut additions: Vec<(Hash, Option<Vec<u8>>)> = Vec::new();

                for child in node_children {
                    let child_hash = self.store_node(child.clone());
                    let separator = self.get_first_key(child);
                    additions.push((child_hash, separator));
                }

                let prev_node: &mut Node = internal_nodes.last_mut().unwrap();
                for (child_hash, separator) in additions {
                    prev_node.values.push(child_hash);
                    if let Some(sep) = separator {
                        if prev_node.keys.len() < prev_node.values.len() - 1 {
                            prev_node.keys.push(sep);
                        }
                    }
                }

                child_idx += node_size;
                continue;
            }

            // Create new internal node
            let mut internal_node = Node::new_internal();
            let node_children = &children[child_idx..child_idx + node_size];

            for (i, child) in node_children.iter().enumerate() {
                let child_hash = self.store_node(child.clone());
                internal_node.values.push(child_hash);

                // Add separator key (first key of next child)
                if i < node_children.len() - 1 {
                    let next_child = &node_children[i + 1];
                    if let Some(separator) = self.get_first_key(next_child) {
                        internal_node.keys.push(separator);
                    }
                }
            }

            internal_nodes.push(internal_node);
            child_idx += node_size;
        }

        // If we only created one internal node, return it directly
        if internal_nodes.len() == 1 {
            let node = internal_nodes.into_iter().next().unwrap();
            if node.values.len() == 1 {
                // Unwrap single-child internal node
                let child_hash = &node.values[0];
                if let Some(child) = self.store.get_node(child_hash) {
                    return child;
                }
            }
            return node;
        }

        // Multiple internal nodes - recursively build parent level
        self.build_internal_from_children(internal_nodes)
    }

    /// Merge two sorted lists of (key, value) tuples
    fn merge_sorted(
        &self,
        old_items: &[(Vec<u8>, Vec<u8>)],
        new_items: &[(Vec<u8>, Vec<u8>)],
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < old_items.len() && j < new_items.len() {
            if old_items[i].0 < new_items[j].0 {
                result.push(old_items[i].clone());
                i += 1;
            } else if old_items[i].0 == new_items[j].0 {
                // New value overwrites old
                result.push(new_items[j].clone());
                i += 1;
                j += 1;
            } else {
                result.push(new_items[j].clone());
                j += 1;
            }
        }

        result.extend_from_slice(&old_items[i..]);
        result.extend_from_slice(&new_items[j..]);
        result
    }

    /// Build leaf nodes from sorted items using rolling hash for splits.
    fn build_leaves(&self, items: &[(Vec<u8>, Vec<u8>)]) -> Vec<Node> {
        if items.is_empty() {
            return vec![Node::new_leaf()];
        }

        let mut leaves = Vec::new();
        let mut current_keys = Vec::new();
        let mut current_values = Vec::new();
        let mut roll_hash = self.seed;

        for (i, (key, value)) in items.iter().enumerate() {
            current_keys.push(key.to_vec());
            current_values.push(value.to_vec());

            // Update rolling hash with the key and value bytes
            roll_hash = self.rolling_hash(roll_hash, key);
            roll_hash = self.rolling_hash(roll_hash, value);

            // Split if: (1) have minimum entries AND hash below pattern OR (2) last item
            let has_min = current_keys.len() >= MIN_NODE_SIZE;
            let should_split = (has_min && roll_hash < self.pattern) || (i == items.len() - 1);

            if should_split && !current_keys.is_empty() {
                let mut leaf = Node::new_leaf();
                leaf.keys = current_keys;
                leaf.values = current_values;

                leaves.push(leaf);

                // Reset for next leaf
                current_keys = Vec::new();
                current_values = Vec::new();
                roll_hash = self.seed;
            }
        }

        if leaves.is_empty() {
            vec![Node::new_leaf()]
        } else {
            leaves
        }
    }

    /// Get root hash
    pub fn get_root_hash(&self) -> Hash {
        self.hash_node(&self.root)
    }

    /// Get reference to the store
    pub fn store(&self) -> &Arc<dyn BlockStore> {
        &self.store
    }

    /// Iterate through all items in the tree with optional prefix filter
    ///
    /// # Arguments
    ///
    /// * `prefix` - Optional prefix to filter keys. If provided, only returns keys starting with this prefix.
    pub fn items(&self, prefix: Option<&[u8]>) -> Vec<(Vec<u8>, Vec<u8>)> {
        let root_hash = self.get_root_hash();
        let mut cursor = TreeCursor::new(self.store.as_ref(), root_hash, prefix);
        let mut result = Vec::new();

        while let Some((key, value)) = cursor.next() {
            // If we have a prefix, check if we've moved past it
            if let Some(prefix_bytes) = prefix {
                if !key.starts_with(prefix_bytes) {
                    break;
                }
            }
            result.push((key, value));
        }

        result
    }

    /// Count total number of items in the tree
    pub fn count(&self) -> usize {
        self.items(None).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_empty() {
        let tree = ProllyTree::default();
        assert_eq!(tree.count(), 0);
    }

    #[test]
    fn test_tree_single_insert() {
        let mut tree = ProllyTree::default();
        let mutations = vec![(b"key1".to_vec(), b"value1".to_vec())];
        tree.insert_batch(mutations, false);
        assert_eq!(tree.count(), 1);
    }

    #[test]
    fn test_tree_multiple_inserts() {
        let mut tree = ProllyTree::default();
        let mutations = vec![
            (b"a".to_vec(), b"val_a".to_vec()),
            (b"b".to_vec(), b"val_b".to_vec()),
            (b"c".to_vec(), b"val_c".to_vec()),
        ];
        tree.insert_batch(mutations, false);
        assert_eq!(tree.count(), 3);

        let items = tree.items(None);
        assert_eq!(items[0], (b"a".to_vec(), b"val_a".to_vec()));
        assert_eq!(items[1], (b"b".to_vec(), b"val_b".to_vec()));
        assert_eq!(items[2], (b"c".to_vec(), b"val_c".to_vec()));
    }

    #[test]
    fn test_tree_update() {
        let mut tree = ProllyTree::default();

        // Insert initial value
        tree.insert_batch(vec![(b"key".to_vec(), b"value1".to_vec())], false);
        assert_eq!(tree.items(None)[0].1, b"value1");

        // Update value
        tree.insert_batch(vec![(b"key".to_vec(), b"value2".to_vec())], false);
        assert_eq!(tree.count(), 1);
        assert_eq!(tree.items(None)[0].1, b"value2");
    }

    #[test]
    fn test_tree_large_batch() {
        let mut tree = ProllyTree::default();

        // Insert 1000 items
        let mut mutations = Vec::new();
        for i in 0..1000 {
            let key = format!("key{:04}", i).into_bytes();
            let value = format!("value{}", i).into_bytes();
            mutations.push((key, value));
        }

        tree.insert_batch(mutations.clone(), false);
        assert_eq!(tree.count(), 1000);

        // Verify items are sorted
        let items = tree.items(None);
        for (i, (key, _)) in items.iter().enumerate() {
            let expected_key = format!("key{:04}", i).into_bytes();
            assert_eq!(key, &expected_key);
        }
    }

    #[test]
    fn test_rolling_hash_deterministic() {
        let tree = ProllyTree::default();
        let hash1 = tree.rolling_hash(tree.seed, b"test");
        let hash2 = tree.rolling_hash(tree.seed, b"test");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_node_deduplication() {
        let mut tree = ProllyTree::default();

        // Insert same data twice
        tree.insert_batch(vec![(b"key".to_vec(), b"value".to_vec())], false);

        tree.insert_batch(vec![(b"key".to_vec(), b"value".to_vec())], false);
        let stats2 = tree.get_stats().clone();

        // Second insert should reuse nodes
        assert!(stats2.nodes_reused > 0 || stats2.nodes_created == 0);
    }
}
