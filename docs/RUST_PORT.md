# Rust Port of ProllyPy

This is a high-performance Rust implementation of the ProllyTree data structure, ported from the original Python implementation.

## Performance

Initial benchmark results (Release build):

- **10,000 inserts**: ~900,000 inserts/sec (11ms)
- **100,000 inserts**: ~827,000 inserts/sec (121ms)
- **25,000 updates**: ~1,100,000 updates/sec (23ms)

The Rust implementation is significantly faster than the Python version while maintaining the same core algorithms.

## Architecture

The codebase is organized into two crates:

### `prolly-core`

The core library implementing Prolly Trees. This crate is designed to be:
- **Wasm-compatible**: Can be compiled to WebAssembly for browser use
- **Zero-dependency core**: Minimal dependencies for maximum portability
- **Pluggable storage**: Supports in-memory, filesystem, and cached storage backends

Modules:
- `node.rs` - Tree node data structure (leaf and internal nodes)
- `stats.rs` - Statistics tracking for node creation and sizes
- `store.rs` - Storage backends (Memory, Filesystem, Cached)
- `cursor.rs` - Tree traversal cursor with O(log n) seeking
- `tree.rs` - Core ProllyTree implementation with rolling hash splitting
- _(Future)_ `diff.rs`, `commonality.rs`, `store_gc.rs`, `commit_graph_store.rs`, `repo.rs`, `db.rs`, `db_diff.rs`

### `prolly-cli` _(Future)_

Command-line interface and SQLite importer. This crate will provide:
- Git-like CLI for version control operations
- SQLite database import functionality
- Query and diff tools

## Features Implemented

✅ Content-addressed nodes with SHA256 hashing
✅ Rolling hash-based splitting for balanced trees
✅ Incremental batch insert with subtree reuse
✅ Multiple storage backends (Memory, Filesystem, LRU Cache)
✅ Tree cursor for efficient traversal and seeking
✅ Node validation and separator invariant checking
✅ Comprehensive test suite (30 tests passing)

## Running Tests

```bash
cargo test --package prolly-core
```

## Running Benchmarks

```bash
cargo build --release --benches
./target/release/deps/tree_bench-*
```

## Example Usage

```rust
use prolly_core::ProllyTree;

// Create a new tree
let mut tree = ProllyTree::default();

// Insert data
let mutations = vec![
    (b"key1".to_vec(), b"value1".to_vec()),
    (b"key2".to_vec(), b"value2".to_vec()),
];
tree.insert_batch(mutations, false);

// Iterate through all items
for (key, value) in tree.items() {
    println!("{:?} => {:?}", key, value);
}
```

## Next Steps

The following modules are planned for future implementation:

1. **diff.rs** - Efficient tree diffing with subtree skipping
2. **store_gc.rs** - Garbage collection for unreachable nodes
3. **commit_graph_store.rs** - Git-like commit storage
4. **repo.rs** - Version control repository abstraction
5. **db.rs** - Database abstraction layer over ProllyTree
6. **db_diff.rs** - Schema-aware database diffing
7. **CLI and SQLite importer** in the `prolly-cli` crate

## License

MIT
