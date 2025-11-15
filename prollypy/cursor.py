"""
TreeCursor for traversing ProllyTree structures.

Provides a cursor abstraction that traverses trees in sorted key order,
independent of tree structure.
"""

from typing import Optional, Tuple
from .store import Store
from .node import Node


class TreeCursor:
    """
    A cursor for traversing a ProllyTree in sorted key order.

    The cursor abstracts away the tree structure (leaf vs internal nodes)
    and provides a uniform interface for iterating through key-value pairs.
    It also supports peeking at the next hash to enable efficient subtree skipping.
    """

    def __init__(self, store: Store, root_hash: str, seek_to: Optional[str] = None):
        """
        Initialize cursor at the beginning of the tree or at a specific prefix.

        Args:
            store: Storage backend
            root_hash: Root hash of tree to traverse
            seek_to: Optional key/prefix to seek to (default: start at beginning)
        """
        self.store = store
        self.root_hash = root_hash
        # Stack of (node, index) tuples representing current position
        # index points to next unvisited child/entry
        self.stack = []
        # Current key-value pair (None until first next() call)
        self.current = None
        # Initialize by descending to first leaf or seeking to prefix
        if seek_to:
            self._seek(root_hash, seek_to)
        else:
            self._descend_to_first(root_hash)

    def _seek(self, node_hash: str, target: str):
        """
        Seek to the first key >= target in O(log n) time.

        Uses separator invariants documented in Node class:
        - separator keys[i] is the first key in child values[i+1]
        - child values[i] contains keys in range based on separators

        Args:
            node_hash: Starting node hash
            target: Target key/prefix to seek to
        """
        node = self.store.get_node(node_hash)
        if node is None:
            return

        while not node.is_leaf:
            # Internal node: find which child should contain target
            # Using separator semantics: keys[i] = first key in values[i+1]
            child_idx = 0
            for i, separator in enumerate(node.keys):
                if target >= separator:
                    child_idx = i + 1
                else:
                    break

            # Push this node with the child index
            self.stack.append((node, child_idx))

            # Descend into the chosen child
            if child_idx < len(node.values):
                child_hash = node.values[child_idx]
                node = self.store.get_node(child_hash)
                if node is None:
                    return
            else:
                return

        # At a leaf node: find first key >= target
        idx = 0
        for i, key in enumerate(node.keys):
            if isinstance(key, str) and key >= target:
                idx = i
                break
        else:
            # All keys in this leaf are < target
            idx = len(node.keys)

        self.stack.append((node, idx))

    def _descend_to_first(self, node_hash: str):
        """Descend to the leftmost leaf starting from node_hash."""
        node = self.store.get_node(node_hash)
        if node is None:
            return

        while not node.is_leaf:
            # Internal node: push it and descend into first child
            self.stack.append((node, 0))
            if len(node.values) == 0:
                return
            child_hash = node.values[0]
            node = self.store.get_node(child_hash)
            if node is None:
                return

        # At a leaf node, push it with index 0
        self.stack.append((node, 0))

    def peek_next_hash(self) -> Optional[str]:
        """
        Peek at the next subtree hash that will be traversed.

        Returns None if at a leaf or no more subtrees.
        This is used to skip identical subtrees during diff.
        """
        if not self.stack:
            return None

        # Look for the next child hash we'll descend into
        for node, idx in reversed(self.stack):
            if not node.is_leaf and idx < len(node.values):
                return node.values[idx]

        return None

    def next(self) -> Optional[Tuple[str, str]]:
        """
        Advance to the next key-value pair.

        Returns:
            (key, value) tuple, or None if exhausted
        """
        if not self.stack:
            self.current = None
            return None

        # Get current node and index
        node, idx = self.stack[-1]

        if node.is_leaf:
            # At a leaf: return current entry and advance
            if idx < len(node.keys):
                key = node.keys[idx]
                value = node.values[idx]
                self.current = (key, value)

                # Advance index
                self.stack[-1] = (node, idx + 1)

                # If we've exhausted this leaf, pop up
                if idx + 1 >= len(node.keys):
                    self.stack.pop()
                    self._advance_to_next_leaf()

                return self.current
            else:
                # Shouldn't happen, but handle gracefully
                self.stack.pop()
                return self.next()
        else:
            # At internal node: shouldn't happen in normal traversal
            # This means we need to descend to next child
            if idx < len(node.values):
                child_hash = node.values[idx]
                # Descend into this child (don't increment idx yet)
                self._descend_to_first(child_hash)
                return self.next()
            else:
                # Exhausted this internal node
                self.stack.pop()
                return self.next()

    def _advance_to_next_leaf(self):
        """After exhausting a leaf, move to the next leaf."""
        while self.stack:
            node, idx = self.stack[-1]

            if not node.is_leaf:
                # Internal node: we just finished child at idx, try next child at idx+1
                next_idx = idx + 1
                if next_idx < len(node.values):
                    child_hash = node.values[next_idx]
                    # Update index to next_idx
                    self.stack[-1] = (node, next_idx)
                    # Descend into child
                    self._descend_to_first(child_hash)
                    return
                else:
                    # Exhausted this internal node
                    self.stack.pop()
            else:
                # Leaf node that's exhausted
                self.stack.pop()

    def skip_subtree(self, subtree_hash: str):
        """
        Skip over a subtree entirely without visiting its entries.

        Args:
            subtree_hash: Hash of subtree to skip
        """
        # Find this hash in our stack and advance past it
        for i in range(len(self.stack) - 1, -1, -1):
            node, idx = self.stack[i]
            if not node.is_leaf and idx > 0 and idx - 1 < len(node.values):
                if node.values[idx - 1] == subtree_hash:
                    # We just descended into this subtree, need to skip it
                    # Pop everything below and including this level
                    self.stack = self.stack[:i+1]
                    # The index is already advanced, so just continue
                    self._advance_to_next_leaf()
                    return

        # If we can't find it, just continue normally
        self._advance_to_next_leaf()
