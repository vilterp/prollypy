//! TreeCursor for traversing ProllyTree structures.
//!
//! Provides a cursor abstraction that traverses trees in sorted key order,
//! independent of tree structure.

use std::sync::Arc;

use crate::node::Node;
use crate::store::BlockStore;
use crate::Hash;

/// A cursor for traversing a ProllyTree in sorted key order.
///
/// The cursor abstracts away the tree structure (leaf vs internal nodes)
/// and provides a uniform interface for iterating through key-value pairs.
/// It also supports peeking at the next hash to enable efficient subtree skipping.
pub struct TreeCursor<'a> {
    store: &'a dyn BlockStore,
    /// Stack of (node, index) tuples representing current position
    /// index points to next unvisited child/entry
    /// Uses Arc<Node> to avoid cloning nodes during traversal
    stack: Vec<(Arc<Node>, usize)>,
    /// Current key-value pair (None until first next() call)
    current: Option<(Arc<[u8]>, Arc<[u8]>)>,
}

impl<'a> TreeCursor<'a> {
    /// Initialize cursor at the beginning of the tree or at a specific prefix.
    ///
    /// # Arguments
    ///
    /// * `store` - Storage backend
    /// * `root_hash` - Root hash of tree to traverse
    /// * `seek_to` - Optional key/prefix to seek to (default: start at beginning)
    pub fn new(
        store: &'a dyn BlockStore,
        root_hash: Hash,
        seek_to: Option<&[u8]>,
    ) -> Self {
        let mut cursor = TreeCursor {
            store,
            stack: Vec::new(),
            current: None,
        };

        // Initialize by descending to first leaf or seeking to prefix
        if let Some(target) = seek_to {
            cursor.seek(&root_hash, target);
        } else {
            cursor.descend_to_first(&root_hash);
        }

        cursor
    }

    /// Seek to the first key >= target in O(log n) time.
    ///
    /// Uses separator invariants documented in Node class:
    /// - separator keys[i] is the first key in child values[i+1]
    /// - child values[i] contains keys in range based on separators
    ///
    /// # Arguments
    ///
    /// * `node_hash` - Starting node hash
    /// * `target` - Target key/prefix to seek to
    fn seek(&mut self, node_hash: &Hash, target: &[u8]) {
        let mut node = match self.store.get_node(node_hash) {
            Some(n) => n,
            None => return,
        };

        while !node.is_leaf {
            // Internal node: find which child should contain target
            // Using separator semantics: keys[i] = first key in values[i+1]
            let mut child_idx = 0;
            for (i, separator) in node.keys.iter().enumerate() {
                if target.as_ref() >= separator.as_ref() {
                    child_idx = i + 1;
                } else {
                    break;
                }
            }

            // Descend into the chosen child
            let child_hash = if child_idx < node.values.len() {
                node.values[child_idx].to_vec()
            } else {
                // Push this node and return
                self.stack.push((node, child_idx));
                return;
            };

            // Push this node with the child index
            self.stack.push((node, child_idx));

            node = match self.store.get_node(&child_hash) {
                Some(n) => n,
                None => return,
            };
        }

        // At a leaf node: find first key >= target
        let mut idx = node.keys.len(); // Default: all keys < target
        for (i, key) in node.keys.iter().enumerate() {
            if key.as_ref() >= target {
                idx = i;
                break;
            }
        }

        self.stack.push((node, idx));
    }

    /// Descend to the leftmost leaf starting from node_hash.
    fn descend_to_first(&mut self, node_hash: &Hash) {
        let mut node = match self.store.get_node(node_hash) {
            Some(n) => n,
            None => return,
        };

        while !node.is_leaf {
            // Internal node: push it and descend into first child
            if node.values.is_empty() {
                self.stack.push((node, 0));
                return;
            }
            let child_hash = node.values[0].to_vec();
            self.stack.push((node, 0));

            node = match self.store.get_node(&child_hash) {
                Some(n) => n,
                None => return,
            };
        }

        // At a leaf node, push it with index 0
        self.stack.push((node, 0));
    }

    /// Peek at the next subtree hash that will be traversed.
    ///
    /// Returns None if at a leaf or no more subtrees.
    /// This is used to skip identical subtrees during diff.
    pub fn peek_next_hash(&self) -> Option<Hash> {
        if self.stack.is_empty() {
            return None;
        }

        // Look for the next child hash we'll descend into
        for (node, idx) in self.stack.iter().rev() {
            if !node.is_leaf && *idx < node.values.len() {
                return Some(node.values[*idx].to_vec());
            }
        }

        None
    }

    /// Advance to the next key-value pair.
    ///
    /// Returns (key, value) tuple, or None if exhausted
    pub fn next(&mut self) -> Option<(Arc<[u8]>, Arc<[u8]>)> {
        if self.stack.is_empty() {
            self.current = None;
            return None;
        }

        // Get current node and index (Arc clone is cheap)
        let (node, idx) = {
            let (n, i) = self.stack.last()?;
            (Arc::clone(n), *i)
        };

        if node.is_leaf {
            // At a leaf: return current entry and advance
            if idx < node.keys.len() {
                let key = node.keys[idx].clone();
                let value = node.values[idx].clone();
                self.current = Some((key.clone(), value.clone()));

                // Advance index
                let stack_len = self.stack.len();
                self.stack[stack_len - 1].1 = idx + 1;

                // If we've exhausted this leaf, pop up
                if idx + 1 >= node.keys.len() {
                    self.stack.pop();
                    self.advance_to_next_leaf();
                }

                return Some((key, value));
            } else {
                // Shouldn't happen, but handle gracefully
                self.stack.pop();
                return self.next();
            }
        } else {
            // At internal node: shouldn't happen in normal traversal
            // This means we need to descend to next child
            if idx < node.values.len() {
                let child_hash = node.values[idx].to_vec();
                // Descend into this child (don't increment idx yet)
                self.descend_to_first(&child_hash);
                return self.next();
            } else {
                // Exhausted this internal node
                self.stack.pop();
                return self.next();
            }
        }
    }

    /// After exhausting a leaf, move to the next leaf.
    fn advance_to_next_leaf(&mut self) {
        while !self.stack.is_empty() {
            // Arc clone is cheap
            let (node, idx) = {
                let (n, i) = self.stack.last().unwrap();
                (Arc::clone(n), *i)
            };

            if !node.is_leaf {
                // Internal node: we just finished child at idx, try next child at idx+1
                let next_idx = idx + 1;
                if next_idx < node.values.len() {
                    let child_hash = node.values[next_idx].to_vec();
                    // Update index to next_idx
                    let stack_len = self.stack.len();
                    self.stack[stack_len - 1].1 = next_idx;
                    // Descend into child
                    self.descend_to_first(&child_hash);
                    return;
                } else {
                    // Exhausted this internal node
                    self.stack.pop();
                }
            } else {
                // Leaf node that's exhausted
                self.stack.pop();
            }
        }
    }

    /// Skip over a subtree entirely without visiting its entries.
    ///
    /// # Arguments
    ///
    /// * `subtree_hash` - Hash of subtree to skip
    pub fn skip_subtree(&mut self, subtree_hash: &Hash) {
        // Find this hash in our stack and advance past it
        for i in (0..self.stack.len()).rev() {
            let (node, idx) = &self.stack[i];
            if !node.is_leaf && *idx > 0 && idx - 1 < node.values.len() {
                if node.values[idx - 1].as_ref() == subtree_hash.as_slice() {
                    // We just descended into this subtree, need to skip it
                    // Pop everything below and including this level
                    self.stack.truncate(i + 1);
                    // The index is already advanced, so just continue
                    self.advance_to_next_leaf();
                    return;
                }
            }
        }

        // If we can't find it, just continue normally
        self.advance_to_next_leaf();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use crate::Node;
    use sha2::{Digest, Sha256};

    fn hash_node(node: &Node) -> Hash {
        let serialized = bincode::serialize(node).unwrap();
        Sha256::digest(&serialized).to_vec()
    }

    #[test]
    fn test_cursor_single_leaf() {
        let store = MemoryBlockStore::new();

        // Create a simple leaf node
        let mut leaf = Node::new_leaf();
        leaf.keys.push(Arc::from(&b"a"[..]));
        leaf.values.push(Arc::from(&b"val_a"[..]));
        leaf.keys.push(Arc::from(&b"b"[..]));
        leaf.values.push(Arc::from(&b"val_b"[..]));

        let root_hash = hash_node(&leaf);
        store.put_node(&root_hash, leaf);

        // Test traversal
        let mut cursor = TreeCursor::new(&store as &dyn BlockStore, root_hash, None);

        let (k, v) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"a");
        assert_eq!(v.as_ref(), b"val_a");

        let (k, v) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"b");
        assert_eq!(v.as_ref(), b"val_b");

        assert!(cursor.next().is_none());
    }

    #[test]
    fn test_cursor_with_internal_node() {
        let store = MemoryBlockStore::new();

        // Create two leaf nodes
        let mut left_leaf = Node::new_leaf();
        left_leaf.keys.push(Arc::from(&b"a"[..]));
        left_leaf.values.push(Arc::from(&b"val_a"[..]));
        left_leaf.keys.push(Arc::from(&b"b"[..]));
        left_leaf.values.push(Arc::from(&b"val_b"[..]));

        let mut right_leaf = Node::new_leaf();
        right_leaf.keys.push(Arc::from(&b"d"[..]));
        right_leaf.values.push(Arc::from(&b"val_d"[..]));
        right_leaf.keys.push(Arc::from(&b"e"[..]));
        right_leaf.values.push(Arc::from(&b"val_e"[..]));

        let left_hash = hash_node(&left_leaf);
        let right_hash = hash_node(&right_leaf);
        store.put_node(&left_hash, left_leaf);
        store.put_node(&right_hash, right_leaf);

        // Create internal node
        let mut internal = Node::new_internal();
        internal.keys.push(Arc::from(&b"d"[..])); // First key of right child
        internal.values.push(Arc::from(left_hash));
        internal.values.push(Arc::from(right_hash));

        let root_hash = hash_node(&internal);
        store.put_node(&root_hash, internal);

        // Test traversal
        let mut cursor = TreeCursor::new(&store as &dyn BlockStore, root_hash, None);

        let (k, _) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"a");
        let (k, _) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"b");
        let (k, _) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"d");
        let (k, _) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"e");

        assert!(cursor.next().is_none());
    }

    #[test]
    fn test_cursor_seek() {
        let store = MemoryBlockStore::new();

        let mut leaf = Node::new_leaf();
        leaf.keys.push(Arc::from(&b"a"[..]));
        leaf.values.push(Arc::from(&b"val_a"[..]));
        leaf.keys.push(Arc::from(&b"d"[..]));
        leaf.values.push(Arc::from(&b"val_d"[..]));
        leaf.keys.push(Arc::from(&b"g"[..]));
        leaf.values.push(Arc::from(&b"val_g"[..]));

        let root_hash = hash_node(&leaf);
        store.put_node(&root_hash, leaf);

        // Seek to "d"
        let mut cursor = TreeCursor::new(&store as &dyn BlockStore, root_hash, Some(b"d"));

        let (k, _) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"d");
        let (k, _) = cursor.next().unwrap();
        assert_eq!(k.as_ref(), b"g");
        assert!(cursor.next().is_none());
    }
}
