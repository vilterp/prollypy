"""
In-memory storage backend for ProllyTree nodes.
"""

from typing import Optional, Iterator
from ..node import Node
from .protocols import BlockStore


class MemoryBlockStore(BlockStore):
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
