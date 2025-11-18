//! # Prolly Core
//!
//! A Rust implementation of Prolly Trees (probabilistic B-trees) with version control capabilities.
//!
//! This crate provides the core data structures and algorithms for content-addressed,
//! probabilistic tree structures with efficient diffing and incremental updates.

pub mod node;
pub mod stats;
pub mod store;
pub mod cursor;
pub mod tree;
pub mod commit_graph_store;
pub mod db;
pub mod diff;
// pub mod commonality;
// pub mod store_gc;
// pub mod repo;
// pub mod db_diff;

// Re-export commonly used types
pub use node::Node;
pub use stats::Stats;
pub use store::{BlockStore, MemoryBlockStore, FileSystemBlockStore, CachedFSBlockStore};
pub use tree::ProllyTree;
pub use cursor::TreeCursor;
pub use commit_graph_store::{Commit, CommitGraphStore, MemoryCommitGraphStore};
pub use db::{DB, Table};
pub use diff::{Added, Deleted, Differ, DiffEvent, DiffStats, Modified};

/// Hash type used throughout the codebase for content addressing
pub type Hash = Vec<u8>;

/// Result type for prolly operations
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
