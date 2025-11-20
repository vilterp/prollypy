//! Storage backends for ProllyTree nodes.
//!
//! Provides a BlockStore trait and multiple implementations:
//! - MemoryBlockStore: In-memory storage using a HashMap
//! - FileSystemBlockStore: Persistent storage using the filesystem
//! - CachedFSBlockStore: Filesystem storage with LRU cache

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::node::Node;
use crate::stats::Stats;
use crate::Hash;

/// Protocol for node storage backends.
pub trait BlockStore: Send + Sync {
    /// Store a node by its hash.
    fn put_node(&self, node_hash: &Hash, node: Node);

    /// Retrieve a node by its hash. Returns None if not found.
    /// Returns Arc<Node> to avoid cloning nodes on retrieval.
    fn get_node(&self, node_hash: &Hash) -> Option<Arc<Node>>;

    /// Delete a node by its hash. Returns true if deleted, false if not found.
    fn delete_node(&self, node_hash: &Hash) -> bool;

    /// Iterate over all node hashes in the store.
    fn list_nodes(&self) -> Vec<Hash>;

    /// Return the total number of nodes in storage.
    fn count_nodes(&self) -> usize;
}

/// In-memory node storage using a HashMap.
#[derive(Debug, Clone)]
pub struct MemoryBlockStore {
    nodes: Arc<Mutex<HashMap<Hash, Arc<Node>>>>,
}

impl MemoryBlockStore {
    /// Create a new in-memory block store.
    pub fn new() -> Self {
        MemoryBlockStore {
            nodes: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for MemoryBlockStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockStore for MemoryBlockStore {
    fn put_node(&self, node_hash: &Hash, node: Node) {
        let mut nodes = self.nodes.lock().unwrap();
        nodes.insert(node_hash.clone(), Arc::new(node));
    }

    fn get_node(&self, node_hash: &Hash) -> Option<Arc<Node>> {
        let nodes = self.nodes.lock().unwrap();
        nodes.get(node_hash).cloned()
    }

    fn delete_node(&self, node_hash: &Hash) -> bool {
        let mut nodes = self.nodes.lock().unwrap();
        nodes.remove(node_hash).is_some()
    }

    fn list_nodes(&self) -> Vec<Hash> {
        let nodes = self.nodes.lock().unwrap();
        nodes.keys().cloned().collect()
    }

    fn count_nodes(&self) -> usize {
        let nodes = self.nodes.lock().unwrap();
        nodes.len()
    }
}

/// File system-based node storage.
pub struct FileSystemBlockStore {
    base_path: PathBuf,
    stats: Arc<Mutex<Stats>>,
}

impl FileSystemBlockStore {
    /// Initialize filesystem storage.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Directory to store nodes in
    pub fn new(base_path: impl AsRef<Path>) -> io::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;

        Ok(FileSystemBlockStore {
            base_path,
            stats: Arc::new(Mutex::new(Stats::new())),
        })
    }

    /// Get the file path for a node hash.
    fn node_path(&self, node_hash: &Hash) -> PathBuf {
        // Convert bytes to hex string for filesystem path
        let hash_str = hex::encode(node_hash);
        // Use first 2 chars as subdirectory for better filesystem performance
        let subdir = &hash_str[..2];
        let dir_path = self.base_path.join(subdir);
        fs::create_dir_all(&dir_path).ok();
        dir_path.join(hash_str)
    }

    /// Serialize a node using bincode.
    fn serialize_node(&self, node: &Node) -> Vec<u8> {
        bincode::serialize(node).expect("Failed to serialize node")
    }

    /// Deserialize a node from bincode.
    fn deserialize_node(&self, data: &[u8]) -> Node {
        bincode::deserialize(data).expect("Failed to deserialize node")
    }

    /// Get node size statistics.
    pub fn get_size_stats(&self) -> crate::stats::SizeStats {
        let stats = self.stats.lock().unwrap();
        stats.get_size_stats()
    }

    /// Get creation statistics.
    pub fn get_creation_stats(&self) -> crate::stats::CreationStats {
        let stats = self.stats.lock().unwrap();
        stats.get_creation_stats()
    }

    /// Print size distributions.
    pub fn print_distributions(&self, bucket_count: usize) {
        let stats = self.stats.lock().unwrap();
        stats.print_distributions(bucket_count);
    }
}

impl BlockStore for FileSystemBlockStore {
    fn put_node(&self, node_hash: &Hash, node: Node) {
        let path = self.node_path(node_hash);
        let serialized = self.serialize_node(&node);

        // Track node size using Stats
        let size = serialized.len();
        {
            let mut stats = self.stats.lock().unwrap();
            if node.is_leaf {
                stats.record_new_leaf(size);
            } else {
                stats.record_new_internal(size);
            }
        }

        let mut file = fs::File::create(path).expect("Failed to create node file");
        file.write_all(&serialized)
            .expect("Failed to write node file");
    }

    fn get_node(&self, node_hash: &Hash) -> Option<Arc<Node>> {
        let path = self.node_path(node_hash);
        if !path.exists() {
            return None;
        }
        let mut file = fs::File::open(path).ok()?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).ok()?;
        Some(Arc::new(self.deserialize_node(&data)))
    }

    fn delete_node(&self, node_hash: &Hash) -> bool {
        let path = self.node_path(node_hash);
        if path.exists() {
            fs::remove_file(path).ok();
            true
        } else {
            false
        }
    }

    fn list_nodes(&self) -> Vec<Hash> {
        let mut nodes = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                let subdir_path = entry.path();
                if subdir_path.is_dir() {
                    if let Ok(subdir_entries) = fs::read_dir(&subdir_path) {
                        for subentry in subdir_entries.flatten() {
                            let file_path = subentry.path();
                            if file_path.is_file() {
                                if let Some(filename) = file_path.file_name() {
                                    if let Some(filename_str) = filename.to_str() {
                                        if let Ok(hash) = hex::decode(filename_str) {
                                            nodes.push(hash);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        nodes
    }

    fn count_nodes(&self) -> usize {
        self.list_nodes().len()
    }
}

/// Filesystem storage with LRU cache for frequently accessed nodes.
pub struct CachedFSBlockStore {
    fs_store: FileSystemBlockStore,
    cache_size: usize,
    cache: Arc<Mutex<LruCache>>,
}

impl CachedFSBlockStore {
    /// Initialize cached filesystem storage.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Directory to store nodes in
    /// * `cache_size` - Maximum number of nodes to keep in memory cache
    pub fn new(base_path: impl AsRef<Path>, cache_size: usize) -> io::Result<Self> {
        Ok(CachedFSBlockStore {
            fs_store: FileSystemBlockStore::new(base_path)?,
            cache_size,
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
        })
    }

    /// Get cache statistics.
    pub fn get_cache_stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let total_requests = cache.cache_hits + cache.cache_misses;
        let hit_rate = if total_requests > 0 {
            (cache.cache_hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        CacheStats {
            cache_size: cache.items.len(),
            max_cache_size: self.cache_size,
            cache_hits: cache.cache_hits,
            cache_misses: cache.cache_misses,
            cache_evictions: cache.cache_evictions,
            hit_rate,
        }
    }

    /// Get size statistics from underlying filesystem store.
    pub fn get_size_stats(&self) -> crate::stats::SizeStats {
        self.fs_store.get_size_stats()
    }

    /// Get creation statistics from underlying filesystem store.
    pub fn get_creation_stats(&self) -> crate::stats::CreationStats {
        self.fs_store.get_creation_stats()
    }

    /// Print size distributions.
    pub fn print_distributions(&self, bucket_count: usize) {
        self.fs_store.print_distributions(bucket_count);
    }
}

impl BlockStore for CachedFSBlockStore {
    fn put_node(&self, node_hash: &Hash, node: Node) {
        // Check if already in cache - if so, no need to write to filesystem
        {
            let cache = self.cache.lock().unwrap();
            if cache.contains(node_hash) {
                // Already have this node, just refresh it in cache
                drop(cache);
                let mut cache = self.cache.lock().unwrap();
                cache.put(node_hash.clone(), Arc::new(node));
                return;
            }
        }

        // Not in cache - write to filesystem first (tracks size)
        self.fs_store.put_node(node_hash, node.clone());

        // Add to cache (will evict if needed)
        let mut cache = self.cache.lock().unwrap();
        cache.put(node_hash.clone(), Arc::new(node));
    }

    fn get_node(&self, node_hash: &Hash) -> Option<Arc<Node>> {
        // Try cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(node) = cache.get(node_hash) {
                return Some(node);
            }
        }

        // Cache miss - read from filesystem
        {
            let mut cache = self.cache.lock().unwrap();
            cache.cache_misses += 1;
        }

        let node = self.fs_store.get_node(node_hash)?;

        // Add to cache for future access
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(node_hash.clone(), node.clone());
        }

        Some(node)
    }

    fn delete_node(&self, node_hash: &Hash) -> bool {
        // Remove from cache if present
        {
            let mut cache = self.cache.lock().unwrap();
            cache.remove(node_hash);
        }

        // Remove from filesystem
        self.fs_store.delete_node(node_hash)
    }

    fn list_nodes(&self) -> Vec<Hash> {
        self.fs_store.list_nodes()
    }

    fn count_nodes(&self) -> usize {
        self.fs_store.count_nodes()
    }
}

/// LRU cache implementation using a HashMap and a linked list simulation
struct LruCache {
    items: HashMap<Hash, (Arc<Node>, usize)>, // (node, timestamp)
    next_timestamp: usize,
    max_size: usize,
    cache_hits: usize,
    cache_misses: usize,
    cache_evictions: usize,
}

impl LruCache {
    fn new(max_size: usize) -> Self {
        LruCache {
            items: HashMap::new(),
            next_timestamp: 0,
            max_size,
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
        }
    }

    fn contains(&self, key: &Hash) -> bool {
        self.items.contains_key(key)
    }

    fn get(&mut self, key: &Hash) -> Option<Arc<Node>> {
        if let Some((node, _)) = self.items.get_mut(key) {
            self.cache_hits += 1;
            let node = node.clone();
            // Update timestamp to mark as recently used
            let timestamp = self.next_timestamp;
            self.next_timestamp += 1;
            self.items.insert(key.clone(), (node.clone(), timestamp));
            Some(node)
        } else {
            None
        }
    }

    fn put(&mut self, key: Hash, value: Arc<Node>) {
        let timestamp = self.next_timestamp;
        self.next_timestamp += 1;

        if self.items.contains_key(&key) {
            // Update existing item
            self.items.insert(key, (value, timestamp));
        } else {
            // Add new item
            if self.items.len() >= self.max_size {
                // Evict oldest item (lowest timestamp)
                if let Some((&ref oldest_key, _)) = self
                    .items
                    .iter()
                    .min_by_key(|(_, (_, timestamp))| timestamp)
                {
                    let oldest_key = oldest_key.clone();
                    self.items.remove(&oldest_key);
                    self.cache_evictions += 1;
                }
            }
            self.items.insert(key, (value, timestamp));
        }
    }

    fn remove(&mut self, key: &Hash) {
        self.items.remove(key);
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: usize,
    pub max_cache_size: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_evictions: usize,
    pub hit_rate: f64,
}

/// Create a block store from a specification string.
///
/// # Arguments
///
/// * `spec` - Store specification, one of:
///   - `:memory:` - in-memory storage
///   - `file:///path/to/dir` - filesystem storage
///   - `cached-file:///path/to/dir` - cached filesystem storage
/// * `cache_size` - Cache size for cached stores (default: 1000)
///
/// # Returns
///
/// BlockStore instance
pub fn create_store_from_spec(
    spec: &str,
    cache_size: Option<usize>,
) -> io::Result<Box<dyn BlockStore>> {
    if spec == ":memory:" {
        Ok(Box::new(MemoryBlockStore::new()))
    } else if let Some(path) = spec.strip_prefix("cached-file://") {
        Ok(Box::new(CachedFSBlockStore::new(
            path,
            cache_size.unwrap_or(1000),
        )?))
    } else if let Some(path) = spec.strip_prefix("file://") {
        Ok(Box::new(FileSystemBlockStore::new(path)?))
    } else if spec.starts_with("s3://") {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "S3 storage not yet implemented",
        ))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Invalid store spec: {}", spec),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use tempfile::TempDir;

    #[test]
    fn test_memory_store() {
        let store = MemoryBlockStore::new();
        let node = Node::new_leaf();
        let hash = vec![1, 2, 3];

        store.put_node(&hash, node.clone());
        let retrieved = store.get_node(&hash);
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_leaf);

        assert_eq!(store.count_nodes(), 1);

        let deleted = store.delete_node(&hash);
        assert!(deleted);
        assert_eq!(store.count_nodes(), 0);
    }

    #[test]
    fn test_filesystem_store() {
        let temp_dir = TempDir::new().unwrap();
        let store = FileSystemBlockStore::new(temp_dir.path()).unwrap();

        let mut node = Node::new_leaf();
        node.keys.push(b"key1".to_vec());
        node.values.push(b"value1".to_vec());
        let hash = vec![1, 2, 3, 4];

        store.put_node(&hash, node.clone());
        let retrieved = store.get_node(&hash);
        assert!(retrieved.is_some());
        let retrieved_node = retrieved.unwrap();
        assert!(retrieved_node.is_leaf);
        assert_eq!(retrieved_node.keys, vec![b"key1".to_vec()]);

        assert_eq!(store.count_nodes(), 1);

        let deleted = store.delete_node(&hash);
        assert!(deleted);
        assert_eq!(store.count_nodes(), 0);
    }

    #[test]
    fn test_cached_store() {
        let temp_dir = TempDir::new().unwrap();
        let store = CachedFSBlockStore::new(temp_dir.path(), 2).unwrap();

        let node1 = Node::new_leaf();
        let node2 = Node::new_internal();
        let hash1 = vec![1, 2, 3];
        let hash2 = vec![4, 5, 6];

        // Put both nodes
        store.put_node(&hash1, node1.clone());
        store.put_node(&hash2, node2.clone());

        // Get node1 - should be cache hit
        let _ = store.get_node(&hash1);
        let stats = store.get_cache_stats();
        assert!(stats.cache_hits > 0);

        // Get non-existent node - should be cache miss
        let _ = store.get_node(&vec![99, 99, 99]);
        let stats = store.get_cache_stats();
        assert!(stats.cache_misses > 0);
    }

    #[test]
    fn test_lru_eviction() {
        let temp_dir = TempDir::new().unwrap();
        let store = CachedFSBlockStore::new(temp_dir.path(), 2).unwrap();

        let node1 = Node::new_leaf();
        let node2 = Node::new_leaf();
        let node3 = Node::new_leaf();
        let hash1 = vec![1];
        let hash2 = vec![2];
        let hash3 = vec![3];

        // Fill cache
        store.put_node(&hash1, node1);
        store.put_node(&hash2, node2);

        // Access hash1 to make it recently used
        let _ = store.get_node(&hash1);

        // Add hash3 - should evict hash2 (least recently used)
        store.put_node(&hash3, node3);

        let stats = store.get_cache_stats();
        assert_eq!(stats.cache_size, 2);
        assert!(stats.cache_evictions > 0);
    }

    #[test]
    fn test_create_store_from_spec() {
        let store = create_store_from_spec(":memory:", None).unwrap();
        let node = Node::new_leaf();
        let hash = vec![1, 2, 3];
        store.put_node(&hash, node);
        assert_eq!(store.count_nodes(), 1);
    }
}
