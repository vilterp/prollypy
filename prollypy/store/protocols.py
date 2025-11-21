"""
Protocol definitions for storage backends.
"""

from typing import Protocol, Optional, Iterator, List
from ..node import Node


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
