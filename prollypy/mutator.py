"""
TreeMutator for incrementally rebuilding ProllyTrees with mutations.

This module provides a streaming approach to tree mutation that:
- Uses TreeCursor to traverse the old tree
- Merges old entries with mutations on-the-fly
- Avoids materializing all items into memory
- Reuses unmodified subtrees when possible
"""

from typing import Iterator, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .tree import ProllyTree
    from .node import Node

from .cursor import TreeCursor


class TreeMutator:
    """
    Handles incremental tree rebuilding with mutations using streaming merge.

    This class maintains a TreeCursor to traverse the old tree and merges
    entries with mutations in a streaming fashion, avoiding the need to
    materialize all entries into memory.
    """

    def __init__(self, tree: 'ProllyTree', root: 'Node', mutations: list[tuple[str, str]]):
        """
        Initialize mutator for rebuilding tree with mutations.

        Args:
            tree: ProllyTree instance (for accessing store, pattern, etc.)
            root: Root node of the tree to rebuild
            mutations: Sorted list of (key, value) tuples to merge in
        """
        self.tree = tree
        self.mutations = mutations
        self.mut_idx = 0

        # Initialize cursor for traversing old tree
        if root.is_leaf and not root.keys:
            # Empty root - no cursor needed
            self.cursor = None
        else:
            # Store root temporarily if needed for cursor traversal
            root_hash = tree._hash_node(root)
            if not root.is_leaf and tree._get_node(root_hash) is None:
                tree.store.put_node(root_hash, root)

            self.cursor = TreeCursor(tree.store, root_hash)

    def rebuild(self) -> 'Node':
        """
        Rebuild tree with mutations using streaming merge.

        Returns:
            New root node with mutations applied
        """
        # Stream merged entries through leaf builder
        merged_stream = self._merge_stream()
        leaves = self.tree._build_leaves(merged_stream)

        if len(leaves) == 1:
            return leaves[0]
        else:
            return self.tree._build_internal_from_children(leaves)

    def _merge_stream(self) -> Iterator[Tuple[str, str]]:
        """
        Generator that yields merged entries from old tree and mutations.

        Performs a streaming merge of the old tree (via cursor) and new
        mutations, similar to merge sort. Mutations overwrite old values
        for duplicate keys.

        Yields:
            (key, value) tuples in sorted order
        """
        # Get first entry from cursor (or None if empty tree)
        current = self.cursor.next() if self.cursor else None

        # Merge old entries with mutations
        while current is not None and self.mut_idx < len(self.mutations):
            key, value = current
            mut_key, mut_value = self.mutations[self.mut_idx]

            if key < mut_key:
                # Old entry comes first
                yield (key, value)
                current = self.cursor.next() if self.cursor else None
            elif key == mut_key:
                # Mutation overwrites old value
                yield (mut_key, mut_value)
                current = self.cursor.next() if self.cursor else None
                self.mut_idx += 1
            else:
                # Mutation comes first
                yield (mut_key, mut_value)
                self.mut_idx += 1

        # Yield remaining old entries
        while current is not None and self.cursor is not None:
            yield current
            current = self.cursor.next()

        # Yield remaining mutations
        while self.mut_idx < len(self.mutations):
            yield self.mutations[self.mut_idx]
            self.mut_idx += 1
