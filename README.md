# ProllyTree Python Prototype

A Python implementation of ProllyTree with pluggable storage backends.

## File Organization

- **`store.py`** - Storage protocol and implementations
  - `Store` - Protocol defining storage interface
  - `Node` - Tree node class
  - `MemoryStore` - In-memory storage implementation
  - `FileSystemStore` - Filesystem-based persistent storage
  - `CachedFSStore` - Filesystem storage with LRU cache
  - `create_store_from_spec()` - Create store from spec string

- **`tree.py`** - Core ProllyTree implementation
  - `ProllyTree` - Main tree class with rolling hash-based splitting
  - Incremental batch insert with subtree reuse
  - Content-addressed nodes

- **`test_tree.py`** - Test suite
  - Basic tests for tree operations
  - Subtree reuse verification

- **`sqlite_import.py`** - SQLite database import tool
  - Import SQLite databases to ProllyTree
  - Configurable storage backend
  - Progress tracking

- **`prolly_tree.py`** - Compatibility layer (re-exports from other modules)

## Usage

### Basic Usage

```python
from tree import ProllyTree
from store import MemoryStore, FileSystemStore, CachedFSStore

# In-memory tree
tree = ProllyTree(pattern=0.0001, seed=42)
tree.insert_batch([(1, 'a'), (2, 'b'), (3, 'c')], verbose=False)
result = tree.verify()

# Filesystem-backed tree
store = FileSystemStore('/tmp/my_tree')
tree = ProllyTree(pattern=0.0001, seed=42, store=store)
tree.insert_batch([(1, 'a'), (2, 'b')], verbose=False)

# Cached filesystem tree (faster reads)
store = CachedFSStore('/tmp/my_tree', cache_size=1000)
tree = ProllyTree(pattern=0.0001, seed=42, store=store)
tree.insert_batch([(1, 'a'), (2, 'b')], verbose=False)

# Check cache performance
stats = store.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")
```

### SQLite Import

```bash
# Import to memory
python sqlite_import.py database.sqlite

# Import to filesystem
python sqlite_import.py database.sqlite --store file:///tmp/prolly_data

# Import to cached filesystem (better performance)
python sqlite_import.py database.sqlite --store cached-file:///tmp/prolly_data --cache-size 500

# Custom parameters
python sqlite_import.py database.sqlite --pattern 0.0001 --seed 42 --batch-size 1000
```

### Running Tests

```bash
python test_tree.py
```

## Store Specifications

The `create_store_from_spec()` function accepts these formats:

- `:memory:` - In-memory storage (default)
- `file:///path/to/dir` - Filesystem storage
- `cached-file:///path/to/dir` - Cached filesystem storage with LRU cache
- `s3://bucket-name` - S3 storage (not yet implemented)

## Key Concepts

### Rolling Hash

The tree uses a rolling hash (Rabin fingerprinting) for deterministic, content-based node splitting. The hash accumulates over all data in a node, and when it falls below the pattern threshold, the node is split.

### Content Addressing

Nodes are identified by the SHA-256 hash of their contents. This enables:
- Structural sharing between tree versions
- Deduplication of identical nodes
- Efficient incremental updates

### Incremental Batch Insert

When inserting a batch of mutations:
1. Partition mutations by subtree ranges
2. Recursively rebuild only affected subtrees
3. Reuse unchanged subtrees by reference

This provides O(k log n) performance where k is the number of mutations, rather than O(n) for a full rebuild.

### LRU Cache (CachedFSStore)

The `CachedFSStore` combines filesystem persistence with an in-memory LRU (Least Recently Used) cache:

**Write behavior:**
- Writes go to both filesystem and cache
- Cache evicts oldest entries when full

**Read behavior:**
- Check cache first (fast)
- On cache miss, read from filesystem and add to cache
- Move accessed items to end (mark as recently used)

**Benefits:**
- Faster reads for frequently accessed nodes
- Reduced filesystem I/O
- Configurable cache size to balance memory usage
- Statistics tracking (hit rate, cache size, etc.)

## Performance

Based on testing with BC00ALL-26SP.sqlite (398,458 rows):
- Initial inserts: ~15,000 rows/sec
- Later inserts: ~2,500-3,000 rows/sec (as tree grows)
- Total nodes created: ~414 for 400k entries
- Pattern 0.0001 results in larger nodes (~1000 entries each)

Performance degrades as the tree grows (expected for Python implementation). The Rust version would be significantly faster.
