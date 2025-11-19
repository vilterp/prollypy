//! Diff algorithm for ProllyTree.
//!
//! Efficiently computes differences between two trees by skipping identical subtrees
//! based on content hashes.

use crate::cursor::TreeCursor;
use crate::store::BlockStore;
use crate::Hash;

/// A key-value pair was added
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Added {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

/// A key was deleted
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deleted {
    pub key: Vec<u8>,
    pub old_value: Vec<u8>,
}

/// A key's value was modified
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Modified {
    pub key: Vec<u8>,
    pub old_value: Vec<u8>,
    pub new_value: Vec<u8>,
}

/// Diff event representing a change between two trees
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffEvent {
    Added(Added),
    Deleted(Deleted),
    Modified(Modified),
}

/// Statistics from a diff operation
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    pub subtrees_skipped: usize,
    pub nodes_compared: usize,
}

/// Diff two ProllyTree structures with statistics tracking
pub struct Differ<'a> {
    store: &'a dyn BlockStore,
    stats: DiffStats,
    prefix: Option<Vec<u8>>,
}

impl<'a> Differ<'a> {
    /// Initialize differ with store.
    ///
    /// # Arguments
    ///
    /// * `store` - Storage backend containing both trees
    pub fn new(store: &'a dyn BlockStore) -> Self {
        Differ {
            store,
            stats: DiffStats::default(),
            prefix: None,
        }
    }

    /// Compute differences between two trees using cursor-based traversal.
    ///
    /// This algorithm handles trees with different structures by comparing
    /// key-value pairs directly, regardless of tree shape.
    ///
    /// When a prefix is provided, uses O(log n) seeking to jump directly to
    /// the prefix location in both trees, avoiding unnecessary iteration.
    ///
    /// # Arguments
    ///
    /// * `old_hash` - Root hash of the old tree
    /// * `new_hash` - Root hash of the new tree
    /// * `prefix` - Optional key prefix to filter diff results
    ///
    /// # Returns
    ///
    /// Vector of DiffEvent objects (Added, Deleted, or Modified) in key order
    pub fn diff(
        &mut self,
        old_hash: &Hash,
        new_hash: &Hash,
        prefix: Option<&[u8]>,
    ) -> Vec<DiffEvent> {
        // Reset stats for new diff operation
        self.stats = DiffStats::default();
        self.prefix = prefix.map(|p| p.to_vec());

        let mut events = Vec::new();

        // If hashes are the same, trees are identical - no diff needed
        if old_hash == new_hash {
            self.stats.subtrees_skipped += 1;
            return events;
        }

        // Create cursors for both trees, seeking to prefix if provided
        let mut old_cursor = TreeCursor::new(self.store, old_hash.clone(), prefix);
        let mut new_cursor = TreeCursor::new(self.store, new_hash.clone(), prefix);

        // Get first entries
        let mut old_entry = old_cursor.next();
        let mut new_entry = new_cursor.next();

        // Track if we've found any matches (for early termination with prefix)
        let mut found_match = false;

        // Merge-like traversal of both trees
        while old_entry.is_some() || new_entry.is_some() {
            // Early termination: if we have a prefix and both entries don't match,
            // and we've already found matches, we're past the prefix range
            if prefix.is_some() {
                let old_matches = old_entry
                    .as_ref()
                    .map(|(k, _)| self.matches_prefix(k))
                    .unwrap_or(false);
                let new_matches = new_entry
                    .as_ref()
                    .map(|(k, _)| self.matches_prefix(k))
                    .unwrap_or(false);

                if found_match && !old_matches && !new_matches {
                    // We've passed the prefix range - stop
                    break;
                }
            }

            // Check for subtree skipping opportunity
            if old_entry.is_some() && new_entry.is_some() {
                let old_next_hash = old_cursor.peek_next_hash();
                let new_next_hash = new_cursor.peek_next_hash();

                if let (Some(old_hash), Some(new_hash)) = (&old_next_hash, &new_next_hash) {
                    if old_hash == new_hash {
                        // Same subtree coming up - skip it!
                        self.stats.subtrees_skipped += 1;
                        old_cursor.skip_subtree(old_hash);
                        new_cursor.skip_subtree(new_hash);
                        old_entry = old_cursor.next();
                        new_entry = new_cursor.next();
                        continue;
                    }
                }
            }

            match (&old_entry, &new_entry) {
                (None, Some((new_key, new_value))) => {
                    // Only new entries remain - all additions
                    if self.matches_prefix(new_key) {
                        found_match = true;
                        events.push(DiffEvent::Added(Added {
                            key: new_key.clone(),
                            value: new_value.clone(),
                        }));
                    } else if prefix.is_some() && found_match {
                        // Past prefix range
                        break;
                    }
                    new_entry = new_cursor.next();
                }
                (Some((old_key, old_value)), None) => {
                    // Only old entries remain - all deletions
                    if self.matches_prefix(old_key) {
                        found_match = true;
                        events.push(DiffEvent::Deleted(Deleted {
                            key: old_key.clone(),
                            old_value: old_value.clone(),
                        }));
                    } else if prefix.is_some() && found_match {
                        // Past prefix range
                        break;
                    }
                    old_entry = old_cursor.next();
                }
                (Some((old_key, old_value)), Some((new_key, new_value))) => {
                    // Both have entries - compare keys
                    if old_key < new_key {
                        // Key only in old tree - deleted
                        if self.matches_prefix(old_key) {
                            found_match = true;
                            events.push(DiffEvent::Deleted(Deleted {
                                key: old_key.clone(),
                                old_value: old_value.clone(),
                            }));
                        }
                        old_entry = old_cursor.next();
                    } else if old_key > new_key {
                        // Key only in new tree - added
                        if self.matches_prefix(new_key) {
                            found_match = true;
                            events.push(DiffEvent::Added(Added {
                                key: new_key.clone(),
                                value: new_value.clone(),
                            }));
                        }
                        new_entry = new_cursor.next();
                    } else {
                        // Same key in both trees
                        if old_value != new_value {
                            // Value changed - modified
                            if self.matches_prefix(old_key) {
                                found_match = true;
                                events.push(DiffEvent::Modified(Modified {
                                    key: old_key.clone(),
                                    old_value: old_value.clone(),
                                    new_value: new_value.clone(),
                                }));
                            }
                        }
                        // else: values are identical, no diff event needed

                        // Advance both cursors
                        old_entry = old_cursor.next();
                        new_entry = new_cursor.next();
                    }
                }
                (None, None) => {
                    // Both exhausted
                    break;
                }
            }
        }

        events
    }

    /// Get statistics from the most recent diff operation
    pub fn get_stats(&self) -> &DiffStats {
        &self.stats
    }

    /// Check if a key matches the prefix filter
    fn matches_prefix(&self, key: &[u8]) -> bool {
        match &self.prefix {
            None => true,
            Some(prefix) => key.starts_with(prefix),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use crate::ProllyTree;
    use std::sync::Arc;

    #[test]
    fn test_diff_empty_trees() {
        let store = Arc::new(MemoryBlockStore::new());
        let tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        let tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));

        let hash1 = tree1.get_root_hash();
        let hash2 = tree2.get_root_hash();

        let mut differ = Differ::new(store.as_ref());
        let events = differ.diff(&hash1, &hash2, None);

        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_diff_additions() {
        let store = Arc::new(MemoryBlockStore::new());

        let tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        let hash1 = tree1.get_root_hash();

        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(
            vec![
                (b"a".to_vec(), b"val_a".to_vec()),
                (b"b".to_vec(), b"val_b".to_vec()),
            ],
            false,
        );
        let hash2 = tree2.get_root_hash();

        let mut differ = Differ::new(store.as_ref());
        let events = differ.diff(&hash1, &hash2, None);

        assert_eq!(events.len(), 2);
        match &events[0] {
            DiffEvent::Added(added) => {
                assert_eq!(added.key, b"a");
                assert_eq!(added.value, b"val_a");
            }
            _ => panic!("Expected Added event"),
        }
    }

    #[test]
    fn test_diff_deletions() {
        let store = Arc::new(MemoryBlockStore::new());

        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(
            vec![
                (b"a".to_vec(), b"val_a".to_vec()),
                (b"b".to_vec(), b"val_b".to_vec()),
            ],
            false,
        );
        let hash1 = tree1.get_root_hash();

        let tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        let hash2 = tree2.get_root_hash();

        let mut differ = Differ::new(store.as_ref());
        let events = differ.diff(&hash1, &hash2, None);

        assert_eq!(events.len(), 2);
        match &events[0] {
            DiffEvent::Deleted(deleted) => {
                assert_eq!(deleted.key, b"a");
                assert_eq!(deleted.old_value, b"val_a");
            }
            _ => panic!("Expected Deleted event"),
        }
    }

    #[test]
    fn test_diff_modifications() {
        let store = Arc::new(MemoryBlockStore::new());

        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(vec![(b"a".to_vec(), b"val_old".to_vec())], false);
        let hash1 = tree1.get_root_hash();

        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(vec![(b"a".to_vec(), b"val_new".to_vec())], false);
        let hash2 = tree2.get_root_hash();

        let mut differ = Differ::new(store.as_ref());
        let events = differ.diff(&hash1, &hash2, None);

        assert_eq!(events.len(), 1);
        match &events[0] {
            DiffEvent::Modified(modified) => {
                assert_eq!(modified.key, b"a");
                assert_eq!(modified.old_value, b"val_old");
                assert_eq!(modified.new_value, b"val_new");
            }
            _ => panic!("Expected Modified event"),
        }
    }

    #[test]
    fn test_diff_with_prefix() {
        let store = Arc::new(MemoryBlockStore::new());

        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(
            vec![
                (b"a/1".to_vec(), b"v1".to_vec()),
                (b"b/1".to_vec(), b"v2".to_vec()),
            ],
            false,
        );
        let hash1 = tree1.get_root_hash();

        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(
            vec![
                (b"a/1".to_vec(), b"v1_new".to_vec()),
                (b"b/1".to_vec(), b"v2".to_vec()),
            ],
            false,
        );
        let hash2 = tree2.get_root_hash();

        let mut differ = Differ::new(store.as_ref());
        let events = differ.diff(&hash1, &hash2, Some(b"a/"));

        // Should only see the modification under "a/" prefix
        assert_eq!(events.len(), 1);
        match &events[0] {
            DiffEvent::Modified(modified) => {
                assert_eq!(modified.key, b"a/1");
            }
            _ => panic!("Expected Modified event"),
        }
    }

    #[test]
    fn test_diff_identical_trees() {
        let store = Arc::new(MemoryBlockStore::new());

        let mut tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree1.insert_batch(vec![(b"a".to_vec(), b"val".to_vec())], false);
        let hash = tree1.get_root_hash();

        let mut differ = Differ::new(store.as_ref());
        let events = differ.diff(&hash, &hash, None);

        assert_eq!(events.len(), 0);
        assert_eq!(differ.get_stats().subtrees_skipped, 1);
    }
}
