"""
Storage backends for ProllyTree nodes.

Provides a BlockStore protocol and multiple implementations:
- MemoryBlockStore: In-memory storage using a dictionary
- FileSystemBlockStore: Persistent storage using the filesystem
- CachedFSBlockStore: Filesystem with LRU cache
- S3BlockStore: S3-based storage for remote push
"""

from typing import Optional

from .protocols import BlockStore, Remote
from .memory import MemoryBlockStore
from .filesystem import FileSystemBlockStore, CachedFSBlockStore
from .s3 import S3BlockStore


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


__all__ = [
    'BlockStore',
    'Remote',
    'MemoryBlockStore',
    'FileSystemBlockStore',
    'CachedFSBlockStore',
    'S3BlockStore',
    'create_store_from_spec',
]
