//! Diff algorithm for ProllyTree.
//!
//! Efficiently computes differences between two trees by skipping identical subtrees
//! based on content hashes. Provides a streaming iterator-based API.

use std::sync::Arc;

use crate::cursor::TreeCursor;
use crate::store::BlockStore;
use crate::Hash;

/// A key-value pair was added
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Added {
    pub key: Arc<[u8]>,
    pub value: Arc<[u8]>,
}

/// A key was deleted
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deleted {
    pub key: Arc<[u8]>,
    pub old_value: Arc<[u8]>,
}

/// A key's value was modified
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Modified {
    pub key: Arc<[u8]>,
    pub old_value: Arc<[u8]>,
    pub new_value: Arc<[u8]>,
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

/// Streaming iterator for diff events between two trees.
///
/// This iterator yields `DiffEvent` values lazily as it traverses the trees,
/// allowing consumers to process results incrementally without accumulating
/// all events in memory.
///
/// # Example
///
/// ```ignore
/// let iter = DiffIterator::new(store, &old_hash, &new_hash, None);
/// for event in iter {
///     match event {
///         DiffEvent::Added(a) => println!("+ {:?}", a.key),
///         DiffEvent::Deleted(d) => println!("- {:?}", d.key),
///         DiffEvent::Modified(m) => println!("M {:?}", m.key),
///     }
/// }
/// // Get stats after iteration
/// let stats = iter.get_stats();
/// ```
pub struct DiffIterator<'a> {
    old_cursor: TreeCursor<'a>,
    new_cursor: TreeCursor<'a>,
    old_entry: Option<(Arc<[u8]>, Arc<[u8]>)>,
    new_entry: Option<(Arc<[u8]>, Arc<[u8]>)>,
    prefix: Option<Vec<u8>>,
    found_match: bool,
    done: bool,
    stats: DiffStats,
}

impl<'a> DiffIterator<'a> {
    /// Create a new diff iterator between two trees.
    ///
    /// # Arguments
    ///
    /// * `store` - Storage backend containing both trees
    /// * `old_hash` - Root hash of the old tree
    /// * `new_hash` - Root hash of the new tree
    /// * `prefix` - Optional key prefix to filter diff results
    pub fn new(
        store: &'a dyn BlockStore,
        old_hash: &Hash,
        new_hash: &Hash,
        prefix: Option<&[u8]>,
    ) -> Self {
        let mut stats = DiffStats::default();

        // If hashes are the same, trees are identical - mark as done immediately
        if old_hash == new_hash {
            stats.subtrees_skipped += 1;
            return DiffIterator {
                old_cursor: TreeCursor::new(store, old_hash.clone(), prefix),
                new_cursor: TreeCursor::new(store, new_hash.clone(), prefix),
                old_entry: None,
                new_entry: None,
                prefix: prefix.map(|p| p.to_vec()),
                found_match: false,
                done: true,
                stats,
            };
        }

        // Create cursors for both trees, seeking to prefix if provided
        let mut old_cursor = TreeCursor::new(store, old_hash.clone(), prefix);
        let mut new_cursor = TreeCursor::new(store, new_hash.clone(), prefix);

        // Get first entries
        let old_entry = old_cursor.next();
        let new_entry = new_cursor.next();

        DiffIterator {
            old_cursor,
            new_cursor,
            old_entry,
            new_entry,
            prefix: prefix.map(|p| p.to_vec()),
            found_match: false,
            done: false,
            stats,
        }
    }

    /// Check if a key matches the prefix filter
    fn matches_prefix(&self, key: &[u8]) -> bool {
        match &self.prefix {
            None => true,
            Some(prefix) => key.starts_with(prefix),
        }
    }

    /// Get statistics from the diff operation.
    ///
    /// Note: Stats are updated as the iterator progresses. Call this after
    /// iteration is complete to get the final statistics.
    pub fn get_stats(&self) -> &DiffStats {
        &self.stats
    }

    /// Consume the iterator and return final statistics.
    pub fn into_stats(self) -> DiffStats {
        self.stats
    }
}

/// Compute differences between two trees, returning a streaming iterator.
///
/// This is the main entry point for diffing trees. It returns an iterator
/// that yields `DiffEvent` values lazily as it traverses the trees.
///
/// # Arguments
///
/// * `store` - Storage backend containing both trees
/// * `old_hash` - Root hash of the old tree
/// * `new_hash` - Root hash of the new tree
/// * `prefix` - Optional key prefix to filter diff results
///
/// # Example
///
/// ```ignore
/// for event in diff(store, &old_hash, &new_hash, None) {
///     match event {
///         DiffEvent::Added(a) => println!("+ {:?}", a.key),
///         DiffEvent::Deleted(d) => println!("- {:?}", d.key),
///         DiffEvent::Modified(m) => println!("M {:?}", m.key),
///     }
/// }
/// ```
pub fn diff<'a>(
    store: &'a dyn BlockStore,
    old_hash: &Hash,
    new_hash: &Hash,
    prefix: Option<&[u8]>,
) -> DiffIterator<'a> {
    DiffIterator::new(store, old_hash, new_hash, prefix)
}

impl<'a> Iterator for DiffIterator<'a> {
    type Item = DiffEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            // Check if we're done
            if self.old_entry.is_none() && self.new_entry.is_none() {
                self.done = true;
                return None;
            }

            // Early termination: if we have a prefix and both entries don't match,
            // and we've already found matches, we're past the prefix range
            if self.prefix.is_some() {
                let old_matches = self
                    .old_entry
                    .as_ref()
                    .map(|(k, _)| self.matches_prefix(k))
                    .unwrap_or(false);
                let new_matches = self
                    .new_entry
                    .as_ref()
                    .map(|(k, _)| self.matches_prefix(k))
                    .unwrap_or(false);

                if self.found_match && !old_matches && !new_matches {
                    // We've passed the prefix range - stop
                    self.done = true;
                    return None;
                }
            }

            // Check for subtree skipping opportunity
            if self.old_entry.is_some() && self.new_entry.is_some() {
                let old_next_hash = self.old_cursor.peek_next_hash();
                let new_next_hash = self.new_cursor.peek_next_hash();

                if let (Some(old_hash), Some(new_hash)) = (&old_next_hash, &new_next_hash) {
                    if old_hash == new_hash {
                        // Same subtree coming up - skip it!
                        self.stats.subtrees_skipped += 1;
                        self.old_cursor.skip_subtree(old_hash);
                        self.new_cursor.skip_subtree(new_hash);
                        self.old_entry = self.old_cursor.next();
                        self.new_entry = self.new_cursor.next();
                        continue;
                    }
                }
            }

            match (&self.old_entry, &self.new_entry) {
                (None, Some((new_key, new_value))) => {
                    // Only new entries remain - all additions
                    if self.matches_prefix(new_key) {
                        self.found_match = true;
                        let event = DiffEvent::Added(Added {
                            key: new_key.clone(),
                            value: new_value.clone(),
                        });
                        self.new_entry = self.new_cursor.next();
                        return Some(event);
                    } else if self.prefix.is_some() && self.found_match {
                        // Past prefix range
                        self.done = true;
                        return None;
                    }
                    self.new_entry = self.new_cursor.next();
                }
                (Some((old_key, old_value)), None) => {
                    // Only old entries remain - all deletions
                    if self.matches_prefix(old_key) {
                        self.found_match = true;
                        let event = DiffEvent::Deleted(Deleted {
                            key: old_key.clone(),
                            old_value: old_value.clone(),
                        });
                        self.old_entry = self.old_cursor.next();
                        return Some(event);
                    } else if self.prefix.is_some() && self.found_match {
                        // Past prefix range
                        self.done = true;
                        return None;
                    }
                    self.old_entry = self.old_cursor.next();
                }
                (Some((old_key, old_value)), Some((new_key, new_value))) => {
                    // Both have entries - compare keys
                    if old_key < new_key {
                        // Key only in old tree - deleted
                        if self.matches_prefix(old_key) {
                            self.found_match = true;
                            let event = DiffEvent::Deleted(Deleted {
                                key: old_key.clone(),
                                old_value: old_value.clone(),
                            });
                            self.old_entry = self.old_cursor.next();
                            return Some(event);
                        }
                        self.old_entry = self.old_cursor.next();
                    } else if old_key > new_key {
                        // Key only in new tree - added
                        if self.matches_prefix(new_key) {
                            self.found_match = true;
                            let event = DiffEvent::Added(Added {
                                key: new_key.clone(),
                                value: new_value.clone(),
                            });
                            self.new_entry = self.new_cursor.next();
                            return Some(event);
                        }
                        self.new_entry = self.new_cursor.next();
                    } else {
                        // Same key in both trees
                        if old_value != new_value {
                            // Value changed - modified
                            if self.matches_prefix(old_key) {
                                self.found_match = true;
                                let event = DiffEvent::Modified(Modified {
                                    key: old_key.clone(),
                                    old_value: old_value.clone(),
                                    new_value: new_value.clone(),
                                });
                                // Advance both cursors
                                self.old_entry = self.old_cursor.next();
                                self.new_entry = self.new_cursor.next();
                                return Some(event);
                            }
                        }
                        // else: values are identical, no diff event needed

                        // Advance both cursors
                        self.old_entry = self.old_cursor.next();
                        self.new_entry = self.new_cursor.next();
                    }
                }
                (None, None) => {
                    // Both exhausted
                    self.done = true;
                    return None;
                }
            }
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

        let events: Vec<_> = diff(store.as_ref(), &hash1, &hash2, None).collect();

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

        let events: Vec<_> = diff(store.as_ref(), &hash1, &hash2, None).collect();

        assert_eq!(events.len(), 2);
        match &events[0] {
            DiffEvent::Added(added) => {
                assert_eq!(added.key.as_ref(), b"a");
                assert_eq!(added.value.as_ref(), b"val_a");
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

        let events: Vec<_> = diff(store.as_ref(), &hash1, &hash2, None).collect();

        assert_eq!(events.len(), 2);
        match &events[0] {
            DiffEvent::Deleted(deleted) => {
                assert_eq!(deleted.key.as_ref(), b"a");
                assert_eq!(deleted.old_value.as_ref(), b"val_a");
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

        let events: Vec<_> = diff(store.as_ref(), &hash1, &hash2, None).collect();

        assert_eq!(events.len(), 1);
        match &events[0] {
            DiffEvent::Modified(modified) => {
                assert_eq!(modified.key.as_ref(), b"a");
                assert_eq!(modified.old_value.as_ref(), b"val_old");
                assert_eq!(modified.new_value.as_ref(), b"val_new");
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

        let events: Vec<_> = diff(store.as_ref(), &hash1, &hash2, Some(b"a/")).collect();

        // Should only see the modification under "a/" prefix
        assert_eq!(events.len(), 1);
        match &events[0] {
            DiffEvent::Modified(modified) => {
                assert_eq!(modified.key.as_ref(), b"a/1");
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

        let mut iter = diff(store.as_ref(), &hash, &hash, None);
        // Exhaust the iterator
        assert!(iter.next().is_none());
        // Check stats
        assert_eq!(iter.get_stats().subtrees_skipped, 1);
    }

    #[test]
    fn test_diff_streaming() {
        // Test that the iterator can be used in streaming fashion
        let store = Arc::new(MemoryBlockStore::new());

        let tree1 = ProllyTree::new(0.01, 42, Some(store.clone()));
        let hash1 = tree1.get_root_hash();

        let mut tree2 = ProllyTree::new(0.01, 42, Some(store.clone()));
        tree2.insert_batch(
            vec![
                (b"a".to_vec(), b"val_a".to_vec()),
                (b"b".to_vec(), b"val_b".to_vec()),
                (b"c".to_vec(), b"val_c".to_vec()),
            ],
            false,
        );
        let hash2 = tree2.get_root_hash();

        let mut iter = diff(store.as_ref(), &hash1, &hash2, None);

        // Get first event
        let first = iter.next();
        assert!(matches!(first, Some(DiffEvent::Added(_))));

        // Get second event
        let second = iter.next();
        assert!(matches!(second, Some(DiffEvent::Added(_))));

        // Get third event
        let third = iter.next();
        assert!(matches!(third, Some(DiffEvent::Added(_))));

        // No more events
        assert!(iter.next().is_none());
    }
}
