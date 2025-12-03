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
use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::cursor::TreeCursor;
use crate::node::Node;
use crate::store::{BlockStore, MemoryBlockStore};
use crate::Hash;

// Chunking constants
const MIN_CHUNK_SIZE: usize = 2;      // Minimum items per node (never split below this)
const MAX_CHUNK_SIZE: usize = 1024;   // Maximum items per node (always split at this)
const TARGET_CHUNK_SIZE: usize = 64;  // Target items per node (threshold doubles every target_size items)

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

    /// Deterministic rolling hash using xxHash (much faster than SHA256).
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
        // Use xxHash3 for fast rolling hash (much faster than SHA256)
        // Combine current hash with data by using current_hash as seed
        let hash64 = xxh3_64_with_seed(data, current_hash as u64);
        hash64 as u32 // Take lower 32 bits
    }

    /// Determine if we should split after this item using size-aware threshold.
    ///
    /// The threshold grows with chunk size: starts at self.pattern, doubles every
    /// TARGET_CHUNK_SIZE items. This ensures:
    /// - Small chunks: low split probability (controlled by pattern)
    /// - Large chunks: increasing split probability
    /// - MAX_CHUNK_SIZE: forced split
    ///
    /// Maintains insertion-order independence since decision only depends on:
    /// - The key's hash (deterministic)
    /// - Current chunk size (same for same key sequence)
    fn should_split(&self, key: &[u8], current_size: usize) -> bool {
        if current_size < MIN_CHUNK_SIZE {
            return false;
        }
        if current_size >= MAX_CHUNK_SIZE {
            return true;
        }

        let key_hash = self.rolling_hash(self.seed, key);

        // Threshold grows exponentially with size
        // At size=TARGET_CHUNK_SIZE, threshold = pattern * 2
        // At size=2*TARGET_CHUNK_SIZE, threshold = pattern * 4
        let size_factor = 2.0_f64.powf(current_size as f64 / TARGET_CHUNK_SIZE as f64);
        let adjusted_threshold = ((self.pattern as f64) * size_factor).min(i32::MAX as f64) as u32;

        key_hash < adjusted_threshold
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
                combined.extend_from_slice(key.as_ref());
                combined.push(b':');
                combined.extend_from_slice(value.as_ref());
                combined.push(b'|');
            }
        } else {
            // Hash separator keys and child hashes
            for (i, child_hash) in node.values.iter().enumerate() {
                if i < node.keys.len() {
                    combined.extend_from_slice(node.keys[i].as_ref());
                } else {
                    combined.push(b'_');
                }
                combined.push(b':');
                combined.extend_from_slice(child_hash.as_ref());
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
        let already_exists = self.store.get_node(&node_hash)
            .map(|opt| opt.is_some())
            .unwrap_or(false);

        if !already_exists {
            // Ignore errors during store (tree operations should not fail mid-operation)
            let _ = self.store.put_node(&node_hash, node.clone());
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
        let result_nodes = self.rebuild_with_mutations(self.root.clone(), &mutations);

        // If we got multiple nodes back, build an internal node structure
        let new_root = if result_nodes.is_empty() {
            Node::new_leaf()
        } else if result_nodes.len() == 1 {
            result_nodes.into_iter().next().unwrap()
        } else {
            self.build_internal_from_children(result_nodes)
        };

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

    /// Collect all key-value pairs from a node's subtree.
    /// Used when we need to rebuild from scratch.
    fn collect_all_items(&self, node: &Node) -> Vec<(Arc<[u8]>, Arc<[u8]>)> {
        if node.is_leaf {
            node.keys
                .iter()
                .zip(node.values.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        } else {
            let mut items = Vec::new();
            for child_hash in node.values.iter() {
                let child_hash_vec = child_hash.to_vec();
                if let Ok(Some(child)) = self.store.get_node(&child_hash_vec) {
                    items.extend(self.collect_all_items(&child));
                }
            }
            items
        }
    }

    /// Core incremental rebuild logic.
    ///
    /// For insertion-order independence, when mutations affect a subtree,
    /// we collect all items from that subtree and rebuild from scratch.
    /// This ensures the same structure regardless of insertion order.
    ///
    /// # Returns
    ///
    /// List of nodes (possibly multiple if the node split)
    fn rebuild_with_mutations(&mut self, node: Node, mutations: &[(Vec<u8>, Vec<u8>)]) -> Vec<Node> {
        if mutations.is_empty() {
            // No mutations for this subtree - REUSE it!
            return vec![node];
        }

        // Collect all existing items from this subtree
        let existing_items = self.collect_all_items(&node);

        // Convert mutations to Arc format and merge
        let mutations_arc: Vec<(Arc<[u8]>, Arc<[u8]>)> = mutations
            .iter()
            .map(|(k, v)| (Arc::from(k.as_slice()), Arc::from(v.as_slice())))
            .collect();

        let merged = self.merge_sorted(&existing_items, &mutations_arc);

        // Rebuild leaves from the merged items
        self.build_leaves(&merged)
    }

    /// Merge two sorted lists of (key, value) pairs
    fn merge_sorted(
        &self,
        old_items: &[(Arc<[u8]>, Arc<[u8]>)],
        new_items: &[(Arc<[u8]>, Arc<[u8]>)],
    ) -> Vec<(Arc<[u8]>, Arc<[u8]>)> {
        let mut result = Vec::with_capacity(old_items.len() + new_items.len());
        let mut i = 0;
        let mut j = 0;

        while i < old_items.len() && j < new_items.len() {
            if old_items[i].0.as_ref() < new_items[j].0.as_ref() {
                result.push(old_items[i].clone());
                i += 1;
            } else if old_items[i].0.as_ref() == new_items[j].0.as_ref() {
                // New value overwrites old
                result.push(new_items[j].clone());
                i += 1;
                j += 1;
            } else {
                result.push(new_items[j].clone());
                j += 1;
            }
        }

        // Append remaining
        while i < old_items.len() {
            result.push(old_items[i].clone());
            i += 1;
        }
        while j < new_items.len() {
            result.push(new_items[j].clone());
            j += 1;
        }

        result
    }

    /// Get the first actual key in a node's subtree.
    fn get_first_key(&self, node: &Node) -> Option<Arc<[u8]>> {
        if node.is_leaf {
            node.keys.first().cloned()
        } else {
            // Internal node - descend to leftmost child
            if node.values.is_empty() {
                return None;
            }
            let child_hash = node.values[0].to_vec();
            let child = self.store.get_node(&child_hash).ok()??;
            self.get_first_key(&child)
        }
    }

    /// Build an internal tree from a list of children using per-child boundary detection.
    ///
    /// Each child independently decides if it's a boundary based on its first key's hash.
    /// A child is a boundary (last child in a node) if hash(seed, first_key) < pattern.
    /// This ensures insertion-order independence since split decisions don't
    /// depend on previous children.
    fn build_internal_from_children(&mut self, children: Vec<Node>) -> Node {
        if children.is_empty() {
            panic!("Cannot build internal node with no children");
        }

        if children.len() == 1 {
            return children[0].clone();
        }

        // Build internal nodes using size-aware boundary detection
        let mut internal_nodes = Vec::new();
        let mut current_children: Vec<&Node> = Vec::new();

        for (i, child) in children.iter().enumerate() {
            current_children.push(child);

            // Get boundary key for this child (first key in subtree)
            let boundary_key = self.get_first_key(child);

            // Use size-aware splitting, or force split on last child
            let is_last = i == children.len() - 1;
            let split = if let Some(ref key) = boundary_key {
                self.should_split(key.as_ref(), current_children.len()) || is_last
            } else {
                is_last // No key, only split if last
            };

            if split && !current_children.is_empty() {
                // Create internal node from current children
                let mut internal_node = Node::new_internal();

                for (j, c) in current_children.iter().enumerate() {
                    let node_hash = self.store_node((*c).clone());
                    internal_node.values.push(Arc::from(node_hash));

                    // Add separator key (first key of next child)
                    if j < current_children.len() - 1 {
                        let next_child = current_children[j + 1];
                        if let Some(separator) = self.get_first_key(next_child) {
                            internal_node.keys.push(separator);
                        }
                    }
                }

                internal_nodes.push(internal_node);

                // Reset for next internal node
                current_children = Vec::new();
            }
        }

        // If we only created one internal node with a single child, unwrap it
        if internal_nodes.len() == 1 {
            let node = internal_nodes.into_iter().next().unwrap();
            if node.values.len() == 1 {
                let child_hash = node.values[0].to_vec();
                if let Ok(Some(child)) = self.store.get_node(&child_hash) {
                    return (*child).clone();
                }
            }
            return node;
        }

        // Multiple internal nodes - recursively build parent level
        self.build_internal_from_children(internal_nodes)
    }

    /// Build leaf nodes from sorted items using per-item boundary detection.
    ///
    /// Each item independently decides if it's a boundary based on its own hash.
    /// An item is a boundary (last item in a node) if hash(seed, key) < pattern.
    /// This ensures insertion-order independence since split decisions don't
    /// depend on previous items.
    fn build_leaves(&self, items: &[(Arc<[u8]>, Arc<[u8]>)]) -> Vec<Node> {
        if items.is_empty() {
            return vec![Node::new_leaf()];
        }

        let mut leaves = Vec::new();
        let mut current_keys: Vec<Arc<[u8]>> = Vec::new();
        let mut current_values: Vec<Arc<[u8]>> = Vec::new();

        for (i, (key, value)) in items.iter().enumerate() {
            current_keys.push(key.clone());
            current_values.push(value.clone());

            // Use size-aware splitting, or force split on last item
            let is_last = i == items.len() - 1;
            let split = self.should_split(key.as_ref(), current_keys.len()) || is_last;

            if split && !current_keys.is_empty() {
                let mut leaf = Node::new_leaf();
                leaf.keys = current_keys;
                leaf.values = current_values;

                leaves.push(leaf);

                // Reset for next leaf
                current_keys = Vec::new();
                current_values = Vec::new();
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
    pub fn items(&self, prefix: Option<&[u8]>) -> Vec<(Arc<[u8]>, Arc<[u8]>)> {
        let root_hash = self.get_root_hash();
        let mut cursor = TreeCursor::new(self.store.as_ref(), root_hash, prefix);
        let mut result = Vec::new();

        while let Some((key, value)) = cursor.next() {
            // If we have a prefix, check if we've moved past it
            if let Some(prefix_bytes) = prefix {
                if !key.as_ref().starts_with(prefix_bytes) {
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
        assert_eq!(items[0].0.as_ref(), b"a");
        assert_eq!(items[0].1.as_ref(), b"val_a");
        assert_eq!(items[1].0.as_ref(), b"b");
        assert_eq!(items[1].1.as_ref(), b"val_b");
        assert_eq!(items[2].0.as_ref(), b"c");
        assert_eq!(items[2].1.as_ref(), b"val_c");
    }

    #[test]
    fn test_tree_update() {
        let mut tree = ProllyTree::default();

        // Insert initial value
        tree.insert_batch(vec![(b"key".to_vec(), b"value1".to_vec())], false);
        assert_eq!(tree.items(None)[0].1.as_ref(), b"value1");

        // Update value
        tree.insert_batch(vec![(b"key".to_vec(), b"value2".to_vec())], false);
        assert_eq!(tree.count(), 1);
        assert_eq!(tree.items(None)[0].1.as_ref(), b"value2");
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
            assert_eq!(key.as_ref(), expected_key.as_slice());
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

    #[test]
    fn test_insertion_order_independence() {
        // Test that inserting data in different batch sizes produces identical trees
        let pattern = 0.5; // High split probability to trigger many splits
        let num_items = 100;

        // Generate all data
        let all_data: Vec<(Vec<u8>, Vec<u8>)> = (0..num_items)
            .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
            .collect();

        // Test 1: Insert all at once
        let mut tree1 = ProllyTree::new(pattern, 42, None);
        tree1.insert_batch(all_data.clone(), false);
        let hash1 = tree1.get_root_hash();

        // Test 2: Insert in two batches
        let mut tree2 = ProllyTree::new(pattern, 42, None);
        let half = num_items / 2;
        let batch1: Vec<(Vec<u8>, Vec<u8>)> = (0..half)
            .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
            .collect();
        let batch2: Vec<(Vec<u8>, Vec<u8>)> = (half..num_items)
            .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
            .collect();
        tree2.insert_batch(batch1, false);
        tree2.insert_batch(batch2, false);
        let hash2 = tree2.get_root_hash();

        // Test 3: Insert in 10 batches
        let mut tree3 = ProllyTree::new(pattern, 42, None);
        let batch_size = num_items / 10;
        for batch_num in 0..10 {
            let start = batch_num * batch_size;
            let end = start + batch_size;
            let batch: Vec<(Vec<u8>, Vec<u8>)> = (start..end)
                .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
                .collect();
            tree3.insert_batch(batch, false);
        }
        let hash3 = tree3.get_root_hash();

        // All should produce identical hashes
        assert_eq!(hash1, hash2, "Single batch vs 2 batches should match");
        assert_eq!(hash1, hash3, "Single batch vs 10 batches should match");
    }

    #[test]
    fn test_insertion_order_independence_minimal() {
        // Minimal test case with just 10 items and high split probability
        let pattern = 0.5;

        let all_data: Vec<(Vec<u8>, Vec<u8>)> = (0..10)
            .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
            .collect();

        // All at once
        let mut tree1 = ProllyTree::new(pattern, 42, None);
        tree1.insert_batch(all_data.clone(), false);
        let hash1 = tree1.get_root_hash();

        // Two batches of 5
        let mut tree2 = ProllyTree::new(pattern, 42, None);
        let batch1: Vec<(Vec<u8>, Vec<u8>)> = (0..5)
            .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
            .collect();
        let batch2: Vec<(Vec<u8>, Vec<u8>)> = (5..10)
            .map(|i| (format!("{:05}", i).into_bytes(), format!("value_{}", i).into_bytes()))
            .collect();
        tree2.insert_batch(batch1, false);
        tree2.insert_batch(batch2, false);
        let hash2 = tree2.get_root_hash();

        assert_eq!(hash1, hash2, "Trees should be identical regardless of batch size");
    }
}
