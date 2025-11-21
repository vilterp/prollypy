"""
Storage backends for ProllyTree nodes.

Provides a BlockStore protocol and multiple implementations:
- MemoryBlockStore: In-memory storage using a dictionary
- FileSystemBlockStore: Persistent storage using the filesystem
"""

from typing import Protocol, Optional, Iterator, List
import os
import pickle
from collections import OrderedDict
from .stats import Stats
from .node import Node

import boto3
from botocore.exceptions import ClientError

import tomllib


class BlockStore(Protocol):
    """Protocol for node storage backends."""

    def put_node(self, node_hash: bytes, node: Node):
        """Store a node by its hash."""
        ...

    def get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve a node by its hash. Returns None if not found."""
        ...

    def url(self) -> str:
        """Return a URL-like identifier for this store."""
        ...

    def delete_node(self, node_hash: bytes) -> bool:
        """Delete a node by its hash. Returns True if deleted, False if not found."""
        ...

    def list_nodes(self) -> Iterator[bytes]:
        """Iterate over all node hashes in the store."""
        ...

    def count_nodes(self) -> int:
        """Return the total number of nodes in storage."""
        ...


class Remote(Protocol):
    """Protocol for remote storage that tracks refs."""

    def list_refs(self) -> List[str]:
        """List all refs on the remote."""
        ...

    def get_ref_commit(self, ref_name: str) -> Optional[str]:
        """Get the commit hash for a ref. Returns None if ref doesn't exist."""
        ...

    def update_ref(self, ref_name: str, old_hash: Optional[str], new_hash: str) -> bool:
        """
        Update a ref with CAS semantics.

        Only updates if the current value matches old_hash.
        old_hash=None means the ref should not exist (create new).

        Returns True if update succeeded, False if there was a conflict.
        """
        ...


class S3BlockStore:
    """S3-based block storage. Currently only supports put_node for pushing."""

    def __init__(self, bucket: str, prefix: str, access_key: str,
                 secret_key: str, region: str = 'us-east-1'):
        """
        Initialize S3 block store.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all blobs
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region (default: us-east-1)
        """
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

    @classmethod
    def from_config(cls, config_path: str, remote_name: str = 'origin') -> 'S3BlockStore':
        """
        Create S3BlockStore from a TOML config file.

        Config format:
            [remotes.origin]
            type = "s3"
            bucket = "your-bucket-name"
            prefix = "prolly/"
            access_key_id = "YOUR_ACCESS_KEY"
            secret_access_key = "YOUR_SECRET_KEY"
            region = "us-east-1"

        Args:
            config_path: Path to TOML config file
            remote_name: Name of the remote to load (default: origin)

        Returns:
            S3BlockStore instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required config fields are missing
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, 'rb') as f:
            config = tomllib.load(f)

        remotes = config.get('remotes', {})
        if remote_name not in remotes:
            raise ValueError(f"Remote '{remote_name}' not found in config")

        remote_config = remotes[remote_name]
        remote_type = remote_config.get('type', 's3')
        if remote_type != 's3':
            raise ValueError(f"Remote '{remote_name}' is not an S3 remote (type: {remote_type})")

        bucket = remote_config.get('bucket')
        prefix = remote_config.get('prefix', '')
        access_key = remote_config.get('access_key_id')
        secret_key = remote_config.get('secret_access_key')
        region = remote_config.get('region', 'us-east-1')

        if not bucket:
            raise ValueError(f"'bucket' not specified for remote '{remote_name}'")
        if not access_key or not secret_key:
            raise ValueError(f"'access_key_id' and 'secret_access_key' required for remote '{remote_name}'")

        return cls(bucket, prefix, access_key, secret_key, region)

    def put_node(self, node_hash: bytes, node: Node):
        """Store a node to S3."""
        data = pickle.dumps(node)
        hash_hex = node_hash.hex()
        key = f"{self.prefix}blocks/{hash_hex[:2]}/{hash_hex[2:]}"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data
        )

    def get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve a node from S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.get_node not yet implemented")

    def url(self) -> str:
        """Return S3 URL for this store."""
        return f"s3://{self.bucket}/{self.prefix}"

    def delete_node(self, node_hash: bytes) -> bool:
        """Delete a node from S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.delete_node not yet implemented")

    def list_nodes(self) -> Iterator[bytes]:
        """List all nodes in S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.list_nodes not yet implemented")

    def count_nodes(self) -> int:
        """Count nodes in S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.count_nodes not yet implemented")

    # Remote protocol methods

    def list_refs(self) -> List[str]:
        """List all refs on the remote."""
        refs = []
        prefix = f"{self.prefix}refs/"

        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                # Extract ref name from key like "prefix/refs/main"
                ref_name = obj['Key'][len(prefix):]
                refs.append(ref_name)

        return refs

    def get_ref_commit(self, ref_name: str) -> Optional[str]:
        """Get the commit hash for a ref. Returns None if ref doesn't exist."""
        key = f"{self.prefix}refs/{ref_name}"

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8').strip()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def update_ref(self, ref_name: str, old_hash: Optional[str], new_hash: str) -> bool:
        """
        Update a ref with CAS semantics using S3 conditional writes.

        Uses ETag for conditional updates to detect concurrent modifications.

        Returns True if update succeeded, False if there was a conflict.
        """
        key = f"{self.prefix}refs/{ref_name}"

        try:
            if old_hash is None:
                # Creating new ref - should not exist
                # Use IfNoneMatch to ensure it doesn't exist
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=new_hash.encode('utf-8'),
                    IfNoneMatch='*'
                )
            else:
                # Updating existing ref - get current ETag first
                try:
                    response = self.s3.get_object(Bucket=self.bucket, Key=key)
                    current_value = response['Body'].read().decode('utf-8').strip()
                    etag = response['ETag']

                    # Check if current value matches expected old_hash
                    if current_value != old_hash:
                        return False

                    # Update with ETag condition
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=new_hash.encode('utf-8'),
                        IfMatch=etag
                    )
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        # Ref doesn't exist but we expected it to
                        return False
                    raise

            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            # PreconditionFailed means ETag didn't match (concurrent update)
            # or object already exists (for IfNoneMatch)
            if error_code in ('PreconditionFailed', 'ConditionalRequestConflict'):
                return False
            raise


class MemoryBlockStore:
    """In-memory node storage using a dictionary."""

    def __init__(self):
        self.nodes: dict[bytes, Node] = {}

    def put_node(self, node_hash: bytes, node: Node):
        """Store a node in memory."""
        self.nodes[node_hash] = node

    def get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve a node from memory."""
        return self.nodes.get(node_hash)

    def delete_node(self, node_hash: bytes) -> bool:
        """Delete a node from memory."""
        if node_hash in self.nodes:
            del self.nodes[node_hash]
            return True
        return False

    def list_nodes(self) -> Iterator[bytes]:
        """Iterate over all node hashes in memory."""
        yield from self.nodes.keys()

    def count_nodes(self) -> int:
        """Return the total number of nodes in storage."""
        return len(self.nodes)

    def url(self) -> str:
        """Return URL for this store."""
        return ":memory:"


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


def create_store_from_spec(spec: str, cache_size: Optional[int] = None) -> BlockStore:
    """
    Create a block store from a specification string.

    Args:
        spec: Store specification, one of:
            - ':memory:' - in-memory storage
            - 'file:///path/to/dir' - filesystem storage
            - 'cached-file:///path/to/dir' - cached filesystem storage
            - 's3://bucket-name' - S3 storage (not yet implemented)
        cache_size: Cache size for cached stores (default: 1000)

    Returns:
        BlockStore instance
    """
    if spec == ':memory:':
        return MemoryBlockStore()
    elif spec.startswith('cached-file://'):
        # Remove 'cached-file://' prefix
        path = spec[14:]
        return CachedFSBlockStore(path, cache_size=cache_size or 1000)
    elif spec.startswith('file://'):
        # Remove 'file://' prefix
        path = spec[7:]
        return FileSystemBlockStore(path)
    elif spec.startswith('s3://'):
        raise NotImplementedError("S3 storage not yet implemented")
    else:
        raise ValueError(f"Invalid store spec: {spec}")
