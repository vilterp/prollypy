//! Diff algorithm for ProllyTree.
//!
//! Efficiently computes differences between two trees by skipping identical subtrees
//! based on content hashes. Provides a streaming iterator-based API using an explicit
//! state machine for clarity.

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

/// Key-value entry type
type Entry = (Arc<[u8]>, Arc<[u8]>);

/// State machine for the diff iterator.
enum DiffState<'a> {
    /// Both trees have remaining entries to compare
    Comparing {
        old_cursor: TreeCursor<'a>,
        new_cursor: TreeCursor<'a>,
        old_entry: Entry,
        new_entry: Entry,
    },
    /// New tree exhausted, draining remaining old entries as deletions
    DrainOld {
        old_cursor: TreeCursor<'a>,
        old_entry: Entry,
    },
    /// Old tree exhausted, draining remaining new entries as additions
    DrainNew {
        new_cursor: TreeCursor<'a>,
        new_entry: Entry,
    },
    /// Iteration complete
    Done,
}

/// Streaming iterator for diff events between two trees.
///
/// This iterator yields `DiffEvent` values lazily as it traverses the trees,
/// allowing consumers to process results incrementally without accumulating
/// all events in memory.
///
/// The iterator uses an explicit state machine with four states:
/// - `Comparing`: Both trees have entries, comparing keys
/// - `DrainOld`: New tree exhausted, emitting deletions
/// - `DrainNew`: Old tree exhausted, emitting additions
/// - `Done`: Iteration complete
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
pub struct DiffIterator<'a> {
    state: DiffState<'a>,
    prefix: Option<Vec<u8>>,
    found_match: bool,
    stats: DiffStats,
}

impl<'a> DiffIterator<'a> {
    /// Create a new diff iterator between two trees.
    pub fn new(
        store: &'a dyn BlockStore,
        old_hash: &Hash,
        new_hash: &Hash,
        prefix: Option<&[u8]>,
    ) -> Self {
        let mut stats = DiffStats::default();

        // If hashes are the same, trees are identical
        if old_hash == new_hash {
            stats.subtrees_skipped += 1;
            return DiffIterator {
                state: DiffState::Done,
                prefix: prefix.map(|p| p.to_vec()),
                found_match: false,
                stats,
            };
        }

        // Create cursors for both trees, seeking to prefix if provided
        let mut old_cursor = TreeCursor::new(store, old_hash.clone(), prefix);
        let mut new_cursor = TreeCursor::new(store, new_hash.clone(), prefix);

        // Get first entries and determine initial state
        let old_entry = old_cursor.next();
        let new_entry = new_cursor.next();

        let state = match (old_entry, new_entry) {
            (Some(old), Some(new)) => DiffState::Comparing {
                old_cursor,
                new_cursor,
                old_entry: old,
                new_entry: new,
            },
            (Some(old), None) => DiffState::DrainOld {
                old_cursor,
                old_entry: old,
            },
            (None, Some(new)) => DiffState::DrainNew {
                new_cursor,
                new_entry: new,
            },
            (None, None) => DiffState::Done,
        };

        DiffIterator {
            state,
            prefix: prefix.map(|p| p.to_vec()),
            found_match: false,
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

    /// Check if we should stop due to passing the prefix range
    fn past_prefix_range(&self, key: &[u8]) -> bool {
        self.prefix.is_some() && self.found_match && !self.matches_prefix(key)
    }

    /// Get statistics from the diff operation.
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
        loop {
            // Take ownership of state to enable state transitions
            let state = std::mem::replace(&mut self.state, DiffState::Done);

            match state {
                DiffState::Done => {
                    return None;
                }

                DiffState::DrainNew {
                    mut new_cursor,
                    new_entry: (key, value),
                } => {
                    // Check prefix early termination
                    if self.past_prefix_range(&key) {
                        return None;
                    }

                    // Advance cursor for next iteration
                    let next_entry = new_cursor.next();
                    self.state = match next_entry {
                        Some(entry) => DiffState::DrainNew {
                            new_cursor,
                            new_entry: entry,
                        },
                        None => DiffState::Done,
                    };

                    // Emit addition if it matches prefix
                    if self.matches_prefix(&key) {
                        self.found_match = true;
                        return Some(DiffEvent::Added(Added { key, value }));
                    }
                    // Otherwise continue to next entry
                }

                DiffState::DrainOld {
                    mut old_cursor,
                    old_entry: (key, old_value),
                } => {
                    // Check prefix early termination
                    if self.past_prefix_range(&key) {
                        return None;
                    }

                    // Advance cursor for next iteration
                    let next_entry = old_cursor.next();
                    self.state = match next_entry {
                        Some(entry) => DiffState::DrainOld {
                            old_cursor,
                            old_entry: entry,
                        },
                        None => DiffState::Done,
                    };

                    // Emit deletion if it matches prefix
                    if self.matches_prefix(&key) {
                        self.found_match = true;
                        return Some(DiffEvent::Deleted(Deleted { key, old_value }));
                    }
                    // Otherwise continue to next entry
                }

                DiffState::Comparing {
                    mut old_cursor,
                    mut new_cursor,
                    old_entry: (old_key, old_value),
                    new_entry: (new_key, new_value),
                } => {
                    // Check prefix early termination
                    if self.prefix.is_some()
                        && self.found_match
                        && !self.matches_prefix(&old_key)
                        && !self.matches_prefix(&new_key)
                    {
                        return None;
                    }

                    // Check for subtree skipping opportunity
                    let old_next_hash = old_cursor.peek_next_hash();
                    let new_next_hash = new_cursor.peek_next_hash();

                    if let (Some(ref old_hash), Some(ref new_hash)) =
                        (&old_next_hash, &new_next_hash)
                    {
                        if old_hash == new_hash {
                            // Same subtree - skip it!
                            self.stats.subtrees_skipped += 1;
                            old_cursor.skip_subtree(old_hash);
                            new_cursor.skip_subtree(new_hash);

                            // Transition to appropriate state
                            let old_next = old_cursor.next();
                            let new_next = new_cursor.next();
                            self.state = match (old_next, new_next) {
                                (Some(old), Some(new)) => DiffState::Comparing {
                                    old_cursor,
                                    new_cursor,
                                    old_entry: old,
                                    new_entry: new,
                                },
                                (Some(old), None) => DiffState::DrainOld {
                                    old_cursor,
                                    old_entry: old,
                                },
                                (None, Some(new)) => DiffState::DrainNew {
                                    new_cursor,
                                    new_entry: new,
                                },
                                (None, None) => DiffState::Done,
                            };
                            continue;
                        }
                    }

                    // Compare keys
                    if old_key < new_key {
                        // Key only in old tree - deletion
                        let next_old = old_cursor.next();
                        self.state = match next_old {
                            Some(entry) => DiffState::Comparing {
                                old_cursor,
                                new_cursor,
                                old_entry: entry,
                                new_entry: (new_key, new_value),
                            },
                            None => DiffState::DrainNew {
                                new_cursor,
                                new_entry: (new_key, new_value),
                            },
                        };

                        if self.matches_prefix(&old_key) {
                            self.found_match = true;
                            return Some(DiffEvent::Deleted(Deleted {
                                key: old_key,
                                old_value,
                            }));
                        }
                    } else if old_key > new_key {
                        // Key only in new tree - addition
                        let next_new = new_cursor.next();
                        self.state = match next_new {
                            Some(entry) => DiffState::Comparing {
                                old_cursor,
                                new_cursor,
                                old_entry: (old_key, old_value),
                                new_entry: entry,
                            },
                            None => DiffState::DrainOld {
                                old_cursor,
                                old_entry: (old_key, old_value),
                            },
                        };

                        if self.matches_prefix(&new_key) {
                            self.found_match = true;
                            return Some(DiffEvent::Added(Added {
                                key: new_key,
                                value: new_value,
                            }));
                        }
                    } else {
                        // Same key - check for modification
                        let is_modified = old_value != new_value;
                        let matches = self.matches_prefix(&old_key);

                        // Advance both cursors
                        let next_old = old_cursor.next();
                        let next_new = new_cursor.next();
                        self.state = match (next_old, next_new) {
                            (Some(old), Some(new)) => DiffState::Comparing {
                                old_cursor,
                                new_cursor,
                                old_entry: old,
                                new_entry: new,
                            },
                            (Some(old), None) => DiffState::DrainOld {
                                old_cursor,
                                old_entry: old,
                            },
                            (None, Some(new)) => DiffState::DrainNew {
                                new_cursor,
                                new_entry: new,
                            },
                            (None, None) => DiffState::Done,
                        };

                        if is_modified && matches {
                            self.found_match = true;
                            return Some(DiffEvent::Modified(Modified {
                                key: old_key,
                                old_value,
                                new_value,
                            }));
                        }
                    }
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
