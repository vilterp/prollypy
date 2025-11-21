"""
Filesystem-based storage backends for ProllyTree nodes.
"""

import os
import pickle
from typing import Optional, Iterator
from collections import OrderedDict

from ..node import Node
from ..stats import Stats


class FileSystemBlockStore:
    """File system-based node storage."""

    def __init__(self, base_path: str):
        """
        Initialize filesystem storage.

        Args:
            base_path: Directory to store nodes in
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        # Statistics tracking
        self.stats = Stats()

    def _node_path(self, node_hash: bytes) -> str:
        """Get the file path for a node hash."""
        # Convert bytes to hex string for filesystem path
        hash_str = node_hash.hex()
        # Use first 2 chars as subdirectory for better filesystem performance
        subdir = hash_str[:2]
        dir_path = os.path.join(self.base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, hash_str)

    def _serialize_node(self, node: Node) -> bytes:
        """Serialize a node using pickle."""
        return pickle.dumps(node)

    def _deserialize_node(self, data: bytes) -> Node:
        """Deserialize a node from pickle."""
        return pickle.loads(data)

    def put_node(self, node_hash: bytes, node: Node):
        """Store a node to filesystem."""
        path = self._node_path(node_hash)
        serialized = self._serialize_node(node)

        # Track node size using Stats
        size = len(serialized)
        if node.is_leaf:
            self.stats.record_new_leaf(size)
        else:
            self.stats.record_new_internal(size)

        with open(path, 'wb') as f:
            f.write(serialized)

    def get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve a node from filesystem."""
        path = self._node_path(node_hash)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return self._deserialize_node(f.read())

    def delete_node(self, node_hash: bytes) -> bool:
        """Delete a node from filesystem."""
        path = self._node_path(node_hash)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_nodes(self) -> Iterator[bytes]:
        """Iterate over all node hashes in filesystem."""
        for subdir in os.listdir(self.base_path):
            subdir_path = os.path.join(self.base_path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    if os.path.isfile(file_path):
                        # Convert hex string filename back to bytes
                        yield bytes.fromhex(filename)

    def count_nodes(self) -> int:
        """Return the total number of nodes in storage."""
        count = 0
        for subdir in os.listdir(self.base_path):
            subdir_path = os.path.join(self.base_path, subdir)
            if os.path.isdir(subdir_path):
                count += len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
        return count

    def get_size_stats(self):
        """Return node size statistics."""
        return self.stats.get_size_stats()

    def url(self) -> str:
        """Return URL for this store."""
        return f"file://{self.base_path}"


class CachedFSBlockStore:
    """Filesystem storage with LRU cache for frequently accessed nodes."""

    def __init__(self, base_path: str, cache_size: int = 1000):
        """
        Initialize cached filesystem storage.

        Args:
            base_path: Directory to store nodes in
            cache_size: Maximum number of nodes to keep in memory cache
        """
        self.fs_store = FileSystemBlockStore(base_path)
        self.cache_size = cache_size
        self.cache: OrderedDict[bytes, Node] = OrderedDict()  # LRU cache using OrderedDict

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def put_node(self, node_hash: bytes, node: Node):
        """Store a node to both cache and filesystem."""
        # Check if already in cache - if so, no need to write to filesystem
        if node_hash in self.cache:
            # Already have this node, just refresh it in cache
            self._cache_put(node_hash, node)
            return

        # Not in cache - write to filesystem first (tracks size)
        self.fs_store.put_node(node_hash, node)

        # Add to cache (will evict if needed)
        self._cache_put(node_hash, node)

    def get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve a node from cache or filesystem."""
        # Try cache first
        node = self._cache_get(node_hash)
        if node is not None:
            self.cache_hits += 1
            return node

        # Cache miss - read from filesystem
        self.cache_misses += 1
        node = self.fs_store.get_node(node_hash)

        if node is not None:
            # Add to cache for future access
            self._cache_put(node_hash, node)

        return node

    def delete_node(self, node_hash: bytes) -> bool:
        """Delete a node from both cache and filesystem."""
        # Remove from cache if present
        if node_hash in self.cache:
            del self.cache[node_hash]

        # Remove from filesystem
        return self.fs_store.delete_node(node_hash)

    def list_nodes(self) -> Iterator[bytes]:
        """Iterate over all node hashes in filesystem."""
        yield from self.fs_store.list_nodes()

    def count_nodes(self) -> int:
        """Return the total number of nodes in storage."""
        return self.fs_store.count_nodes()

    def _cache_get(self, key: bytes) -> Optional[Node]:
        """Get item from cache and move to end (most recently used)."""
        if key not in self.cache:
            return None
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def _cache_put(self, key: bytes, value: Node):
        """Add item to cache, evicting LRU item if at capacity."""
        if key in self.cache:
            # Update existing item and move to end
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new item
            self.cache[key] = value
            # Evict oldest if over capacity
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)  # Remove first (oldest) item
                self.cache_evictions += 1

    def get_cache_stats(self):
        """Return cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_evictions': self.cache_evictions,
            'hit_rate': f"{hit_rate:.1f}%"
        }

    def get_size_stats(self):
        """Return node size statistics from underlying filesystem store."""
        return self.fs_store.get_size_stats()

    def get_creation_stats(self):
        """Return cumulative node creation statistics from underlying filesystem store."""
        return self.fs_store.stats.get_creation_stats()

    def print_distributions(self, bucket_count: int = 10):
        """Print size distributions for leaf and internal nodes."""
        self.fs_store.stats.print_distributions(bucket_count)

    def url(self) -> str:
        """Return URL for this store."""
        return f"cached-file://{self.fs_store.base_path}"
