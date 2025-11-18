//! Tree node implementation for ProllyTree.
//!
//! A Node represents either a leaf or internal node in the tree structure.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::store::BlockStore;

/// Tree node - can be leaf or internal.
///
/// ## Separator Semantics for Internal Nodes
///
/// For internal nodes with n children, there are n-1 separator keys.
/// Each separator key defines the boundary between adjacent children.
///
/// **Key Invariant**: For an internal node:
/// - `keys[i]` is the FIRST key in child `values[i+1]`
/// - Child `values[i]` contains all keys k where:
///   - If i == 0: k < keys[0] (all keys less than first separator)
///   - If 0 < i < len(keys): keys[i-1] <= k < keys[i] (range between separators)
///   - If i == len(keys): k >= keys[-1] (all keys >= last separator)
///
/// **Example**: Internal node with keys=['d', 'h', 'm'] and 4 children:
/// ```text
/// values[0]: all keys k where k < 'd'
/// values[1]: all keys k where 'd' <= k < 'h'
/// values[2]: all keys k where 'h' <= k < 'm'
/// values[3]: all keys k where k >= 'm'
/// ```
///
/// ## Seeking with Separators
///
/// To find which child should contain a target key:
/// ```text
/// child_idx = 0
/// for i, separator in enumerate(node.keys):
///     if target >= separator:
///         child_idx = i + 1
///     else:
///         break
/// # Descend into values[child_idx]
/// ```
///
/// ## Leaf Nodes
///
/// For leaf nodes:
/// - `keys` contains the actual data keys
/// - `values` contains the corresponding data values
/// - len(keys) == len(values)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub is_leaf: bool,
    /// Separator keys (for internal) or actual keys (for leaves)
    pub keys: Vec<Vec<u8>>,
    /// For leaf nodes: data values (bytes)
    /// For internal nodes: child hashes (bytes)
    pub values: Vec<Vec<u8>>,
}

impl Node {
    /// Create a new leaf node
    pub fn new_leaf() -> Self {
        Node {
            is_leaf: true,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create a new internal node
    pub fn new_internal() -> Self {
        Node {
            is_leaf: false,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Validate this node and its entire subtree.
    ///
    /// Recursively traverses all keys in sorted order and checks for duplicates/ordering.
    /// Also validates separator invariants for internal nodes.
    ///
    /// # Arguments
    ///
    /// * `store` - Store to retrieve child nodes (required for internal nodes)
    /// * `context` - Optional context string for error messages
    ///
    /// # Errors
    ///
    /// Returns error if the node or its subtree is invalid
    pub fn validate(
        &self,
        store: Option<&dyn BlockStore>,
        context: &str,
    ) -> Result<(), String> {
        let context_str = if context.is_empty() {
            String::new()
        } else {
            format!(" ({})", context)
        };

        // Basic structural checks
        if self.is_leaf {
            if self.keys.len() != self.values.len() {
                return Err(format!(
                    "Leaf node has {} keys but {} values{}",
                    self.keys.len(),
                    self.values.len(),
                    context_str
                ));
            }
        } else {
            if self.values.len() != self.keys.len() + 1 {
                return Err(format!(
                    "Internal node has {} keys but {} children (should be keys+1){}",
                    self.keys.len(),
                    self.values.len(),
                    context_str
                ));
            }

            // Validate separator invariants for internal nodes
            if let Some(store) = store {
                self.validate_separators(store, &context_str)?;
            }
        }

        // Collect all keys from this subtree in order
        let mut all_keys = Vec::new();
        self.collect_keys(&mut all_keys, store)?;

        // Check that all keys are in sorted order with no duplicates
        let mut prev_key: Option<&Vec<u8>> = None;
        for (i, key) in all_keys.iter().enumerate() {
            if let Some(prev) = prev_key {
                if key == prev {
                    return Err(format!(
                        "Duplicate key at position {}: {:?}{}",
                        i, key, context_str
                    ));
                } else if key < prev {
                    // Print debug info about the node
                    let mut error_msg = format!(
                        "Keys out of order at position {}: {:?} > {:?}{}",
                        i, prev, key, context_str
                    );
                    error_msg.push_str(&format!("\nNode structure:"));
                    error_msg.push_str(&format!("\n  is_leaf: {}", self.is_leaf));
                    error_msg.push_str(&format!("\n  num_keys: {}", self.keys.len()));
                    error_msg.push_str(&format!("\n  num_values: {}", self.values.len()));
                    if !self.is_leaf {
                        error_msg.push_str(&format!("\n  separator_keys: {:?}", self.keys));
                        // Print first key of each child
                        if let Some(store) = store {
                            error_msg.push_str("\n  Children first keys:");
                            for (j, child_hash) in self.values.iter().enumerate() {
                                if let Some(child) = store.get_node(child_hash) {
                                    if !child.keys.is_empty() {
                                        let first_key = &child.keys[0];
                                        error_msg
                                            .push_str(&format!("\n    child[{}]: {:?}", j, first_key));
                                    }
                                }
                            }
                        }
                    } else {
                        let keys_display = if self.keys.len() > 10 {
                            format!("  keys: {:?}...", &self.keys[..10])
                        } else {
                            format!("  keys: {:?}", self.keys)
                        };
                        error_msg.push_str(&format!("\n{}", keys_display));
                    }
                    return Err(error_msg);
                }
            }
            prev_key = Some(key);
        }

        Ok(())
    }

    /// Validate separator invariants for internal nodes.
    ///
    /// Checks that each separator key equals the first key in its corresponding child.
    fn validate_separators(
        &self,
        store: &dyn BlockStore,
        context_str: &str,
    ) -> Result<(), String> {
        for (i, separator) in self.keys.iter().enumerate() {
            // separator should be the first key in child i+1
            let child_idx = i + 1;
            if child_idx >= self.values.len() {
                return Err(format!(
                    "Separator {} points to non-existent child {}{}",
                    i, child_idx, context_str
                ));
            }

            let child_hash = &self.values[child_idx];
            let child = store
                .get_node(child_hash)
                .ok_or_else(|| format!("Child {} not found in store{}", child_idx, context_str))?;

            // Get first key from child
            let first_key = self
                .get_first_key(&child, store)
                .ok_or_else(|| format!("Child {} has no keys{}", child_idx, context_str))?;

            if separator != &first_key {
                return Err(format!(
                    "Separator invariant violated at index {}{}\n\
                     Expected separator: {:?}\n\
                     Actual separator: {:?}\n\
                     Child index: {}",
                    i, context_str, first_key, separator, child_idx
                ));
            }
        }
        Ok(())
    }

    /// Get the first key in a node's subtree.
    fn get_first_key(&self, node: &Node, store: &dyn BlockStore) -> Option<Vec<u8>> {
        if node.is_leaf {
            node.keys.first().cloned()
        } else {
            // Descend to leftmost child
            if node.values.is_empty() {
                return None;
            }
            let child_hash = &node.values[0];
            let child = store.get_node(child_hash)?;
            self.get_first_key(&child, store)
        }
    }

    /// Recursively collect all keys from this node's subtree in traversal order.
    fn collect_keys(
        &self,
        result: &mut Vec<Vec<u8>>,
        store: Option<&dyn BlockStore>,
    ) -> Result<(), String> {
        if self.is_leaf {
            // For leaf nodes, just add all keys
            result.extend(self.keys.iter().cloned());
        } else {
            // For internal nodes, traverse children in order
            let store = store.ok_or("Cannot validate internal node without store")?;
            for child_hash in &self.values {
                let child = store.get_node(child_hash).ok_or_else(|| {
                    format!("Child node {} not found in store", hex::encode(child_hash))
                })?;
                child.collect_keys(result, Some(store))?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_leaf {
            write!(
                f,
                "Leaf({} keys)",
                self.keys.len()
            )
        } else {
            write!(
                f,
                "Internal(keys={}, children={})",
                self.keys.len(),
                self.values.len()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;

    #[test]
    fn test_new_leaf() {
        let node = Node::new_leaf();
        assert!(node.is_leaf);
        assert_eq!(node.keys.len(), 0);
        assert_eq!(node.values.len(), 0);
    }

    #[test]
    fn test_new_internal() {
        let node = Node::new_internal();
        assert!(!node.is_leaf);
        assert_eq!(node.keys.len(), 0);
        assert_eq!(node.values.len(), 0);
    }

    #[test]
    fn test_validate_leaf_mismatch() {
        let mut node = Node::new_leaf();
        node.keys.push(b"key1".to_vec());
        // values is empty, so mismatch
        let result = node.validate(None, "");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Leaf node has 1 keys but 0 values"));
    }

    #[test]
    fn test_validate_internal_mismatch() {
        let mut node = Node::new_internal();
        node.keys.push(b"key1".to_vec());
        node.values.push(b"hash1".to_vec());
        // Should have 2 values for 1 key
        let result = node.validate(None, "");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Internal node has 1 keys but 1 children"));
    }

    #[test]
    fn test_validate_leaf_ordering() {
        let mut node = Node::new_leaf();
        node.keys.push(b"b".to_vec());
        node.values.push(b"val1".to_vec());
        node.keys.push(b"a".to_vec());
        node.values.push(b"val2".to_vec());

        let result = node.validate(None, "");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Keys out of order"));
    }

    #[test]
    fn test_validate_leaf_duplicates() {
        let mut node = Node::new_leaf();
        node.keys.push(b"a".to_vec());
        node.values.push(b"val1".to_vec());
        node.keys.push(b"a".to_vec());
        node.values.push(b"val2".to_vec());

        let result = node.validate(None, "");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Duplicate key"));
    }

    #[test]
    fn test_validate_valid_leaf() {
        let mut node = Node::new_leaf();
        node.keys.push(b"a".to_vec());
        node.values.push(b"val1".to_vec());
        node.keys.push(b"b".to_vec());
        node.values.push(b"val2".to_vec());
        node.keys.push(b"c".to_vec());
        node.values.push(b"val3".to_vec());

        let result = node.validate(None, "");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_separator_invariant() {
        let store = MemoryBlockStore::new();

        // Create two leaf children
        let mut left_leaf = Node::new_leaf();
        left_leaf.keys.push(b"a".to_vec());
        left_leaf.values.push(b"val_a".to_vec());

        let mut right_leaf = Node::new_leaf();
        right_leaf.keys.push(b"d".to_vec());
        right_leaf.values.push(b"val_d".to_vec());

        // Store them
        let left_hash = vec![1, 2, 3];
        let right_hash = vec![4, 5, 6];
        store.put_node(&left_hash, left_leaf);
        store.put_node(&right_hash, right_leaf);

        // Create internal node with correct separator
        let mut internal = Node::new_internal();
        internal.keys.push(b"d".to_vec()); // First key of right child
        internal.values.push(left_hash);
        internal.values.push(right_hash);

        let result = internal.validate(Some(&store as &dyn BlockStore), "");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_separator_invariant_violation() {
        let store = MemoryBlockStore::new();

        // Create two leaf children
        let mut left_leaf = Node::new_leaf();
        left_leaf.keys.push(b"a".to_vec());
        left_leaf.values.push(b"val_a".to_vec());

        let mut right_leaf = Node::new_leaf();
        right_leaf.keys.push(b"d".to_vec());
        right_leaf.values.push(b"val_d".to_vec());

        // Store them
        let left_hash = vec![1, 2, 3];
        let right_hash = vec![4, 5, 6];
        store.put_node(&left_hash, left_leaf);
        store.put_node(&right_hash, right_leaf);

        // Create internal node with WRONG separator
        let mut internal = Node::new_internal();
        internal.keys.push(b"wrong".to_vec()); // Should be "d"
        internal.values.push(left_hash);
        internal.values.push(right_hash);

        let result = internal.validate(Some(&store as &dyn BlockStore), "");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Separator invariant violated"));
    }
}
