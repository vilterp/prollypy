"""
TreeMutator for incrementally rebuilding ProllyTrees with mutations.

This module provides a streaming approach to tree mutation that:
- Uses TreeCursor to traverse the old tree
- Merges old entries with mutations on-the-fly
- Avoids materializing all items into memory
- Reuses unmodified subtrees when possible
"""

from typing import Iterator, Tuple, Union, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .tree import ProllyTree

from .cursor import TreeCursor
from .node import Node


@dataclass
class SubtreeRef:
    """Reference to an existing subtree that can be reused."""
    hash: str


# Type for items in the merge stream
MergeItem = Union[Tuple[str, str], SubtreeRef]


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

    def rebuild(self) -> Node:
        """
        Rebuild tree with mutations using streaming merge.

        Returns:
            New root node with mutations applied
        """
        # Stream merged entries and build children (leaves or reused subtrees)
        merged_stream = self._merge_stream()
        children = self._build_children(merged_stream)

        if len(children) == 1:
            return children[0]
        else:
            return self._build_internal_from_children(children)

    def _build_children(self, stream) -> list[Node]:
        """
        Build child nodes from a stream of entries and subtree reuse markers.

        Handles:
        - (key, value) entries: accumulated into new leaf nodes
        - (REUSE_SUBTREE, hash) markers: fetched and incorporated as-is

        Args:
            stream: Iterator yielding (key, value) or (REUSE_SUBTREE, hash)

        Returns:
            List of child nodes (mix of newly built leaves and reused nodes)
        """
        children = []
        current_keys = []
        current_values = []
        roll_hash = self.tree.seed
        MIN_NODE_SIZE = 2

        for item in stream:
            # Check if this is a subtree reference
            if isinstance(item, SubtreeRef):
                # Flush any pending leaf data
                if current_keys:
                    leaf = Node(is_leaf=True)
                    leaf.keys = current_keys
                    leaf.values = current_values
                    if self.tree.validate:
                        leaf.validate(self.tree.store, context="_build_children")
                    children.append(leaf)
                    current_keys = []
                    current_values = []
                    roll_hash = self.tree.seed

                # Add the reused child
                child_node = self.tree._get_node(item.hash)
                if child_node is None:
                    raise ValueError(f"Referenced subtree {item.hash} not found in store")

                # Mark this node as reused so we don't re-store it
                child_node._reused_hash = item.hash
                children.append(child_node)
                continue

            # Regular entry - add to current leaf
            key, value = item
            current_keys.append(key)
            current_values.append(value)

            # Update rolling hash
            key_bytes = str(key).encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            roll_hash = self.tree._rolling_hash(roll_hash, key_bytes)
            roll_hash = self.tree._rolling_hash(roll_hash, value_bytes)

            # Check if we should split
            has_min = len(current_keys) >= MIN_NODE_SIZE
            should_split = has_min and roll_hash < self.tree.pattern

            if should_split:
                leaf = Node(is_leaf=True)
                leaf.keys = current_keys
                leaf.values = current_values
                if self.tree.validate:
                    leaf.validate(self.tree.store, context="_build_children")
                children.append(leaf)

                # Reset for next leaf
                current_keys = []
                current_values = []
                roll_hash = self.tree.seed

        # Flush any remaining entries
        if current_keys:
            leaf = Node(is_leaf=True)
            leaf.keys = current_keys
            leaf.values = current_values
            if self.tree.validate:
                leaf.validate(self.tree.store, context="_build_children")
            children.append(leaf)

        return children if children else [Node(is_leaf=True)]

    def _merge_stream(self) -> Iterator[MergeItem]:
        """
        Generator that yields merged entries from old tree and mutations.

        Performs a streaming merge of the old tree (via cursor) and new
        mutations, similar to merge sort. Mutations overwrite old values
        for duplicate keys.

        Optimizes by detecting and skipping unchanged subtrees, yielding
        SubtreeRef objects instead of iterating through all their entries.

        Yields:
            - (key, value) tuples for individual entries
            - SubtreeRef objects for unchanged subtrees to reuse
        """
        if not self.cursor:
            # No cursor, just yield all mutations
            while self.mut_idx < len(self.mutations):
                yield self.mutations[self.mut_idx]
                self.mut_idx += 1
            return

        # Check for initial subtree reuse opportunity
        current = self._try_skip_subtree()
        if current is None:
            current = self.cursor.next()

        # Merge old entries with mutations
        while current is not None and self.mut_idx < len(self.mutations):
            # Check if this is a subtree reference
            if isinstance(current, SubtreeRef):
                yield current
                current = self._try_skip_subtree()
                if current is None:
                    current = self.cursor.next() if self.cursor else None
                continue

            key, value = current
            mut_key, mut_value = self.mutations[self.mut_idx]

            if key < mut_key:
                # Old entry comes first
                yield (key, value)
                current = self._try_skip_subtree()
                if current is None:
                    current = self.cursor.next() if self.cursor else None
            elif key == mut_key:
                # Mutation overwrites old value
                yield (mut_key, mut_value)
                current = self._try_skip_subtree()
                if current is None:
                    current = self.cursor.next() if self.cursor else None
                self.mut_idx += 1
            else:
                # Mutation comes first
                yield (mut_key, mut_value)
                self.mut_idx += 1

        # Yield remaining old entries (including subtree references)
        while current is not None and self.cursor is not None:
            if isinstance(current, SubtreeRef):
                yield current
                current = self._try_skip_subtree()
                if current is None:
                    current = self.cursor.next()
            else:
                yield current
                current = self._try_skip_subtree()
                if current is None:
                    current = self.cursor.next()

        # Yield remaining mutations
        while self.mut_idx < len(self.mutations):
            yield self.mutations[self.mut_idx]
            self.mut_idx += 1

    def _try_skip_subtree(self) -> Optional[Union[Tuple[str, str], SubtreeRef]]:
        """
        Check if we can skip the next subtree and reuse it.

        Returns:
            - SubtreeRef if subtree can be reused
            - None if no subtree to skip or it contains mutations
        """
        if not self.cursor:
            return None

        subtree_info = self.cursor.peek_next_subtree()
        if not subtree_info:
            return None

        child_hash, min_key, max_key = subtree_info

        # Check if any remaining mutations fall in this subtree's range
        if self.mut_idx >= len(self.mutations):
            # No more mutations - can skip this and all remaining subtrees
            self.cursor.skip_subtree(child_hash)
            return SubtreeRef(child_hash)

        # Check if next mutation is in this subtree's range
        next_mut_key = self.mutations[self.mut_idx][0]

        # Determine if mutation is in range [min_key, max_key)
        in_range = True
        if min_key is not None and next_mut_key < min_key:
            in_range = False
        if max_key is not None and next_mut_key >= max_key:
            in_range = False

        if not in_range:
            # Next mutation is outside this subtree - can skip it
            self.cursor.skip_subtree(child_hash)
            return SubtreeRef(child_hash)

        # Mutation is in range - can't skip subtree
        return None

    def _build_leaves(self, items) -> list[Node]:
        """
        Build leaf nodes from sorted items using rolling hash for splits.

        Accepts an iterable (list or iterator) of (key, value) tuples and
        builds leaf nodes by streaming through the items. Split points are
        determined by rolling hash being below pattern threshold, with a
        minimum of 2 entries per node to avoid degenerate splits.

        Args:
            items: Iterable of (key, value) tuples in sorted order

        Returns:
            List of leaf Node objects
        """
        MIN_NODE_SIZE = 2  # Minimum entries per node to avoid degenerate trees

        leaves = []
        current_keys = []
        current_values = []
        roll_hash = self.tree.seed  # Start with seed

        for key, value in items:
            current_keys.append(key)
            current_values.append(value)

            # Update rolling hash with the key and value bytes
            key_bytes = str(key).encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            roll_hash = self.tree._rolling_hash(roll_hash, key_bytes)
            roll_hash = self.tree._rolling_hash(roll_hash, value_bytes)

            # Split if we have minimum entries AND hash below pattern
            has_min = len(current_keys) >= MIN_NODE_SIZE
            should_split = has_min and roll_hash < self.tree.pattern

            if should_split:
                leaf = Node(is_leaf=True)
                leaf.keys = current_keys
                leaf.values = current_values

                # Validate leaf before adding
                if self.tree.validate:
                    leaf.validate(self.tree.store, context="_build_leaves")

                leaves.append(leaf)

                # Reset for next leaf
                current_keys = []
                current_values = []
                roll_hash = self.tree.seed  # Reset hash for next node

        # Create final leaf with any remaining items
        if current_keys:
            leaf = Node(is_leaf=True)
            leaf.keys = current_keys
            leaf.values = current_values

            if self.tree.validate:
                leaf.validate(self.tree.store, context="_build_leaves")

            leaves.append(leaf)

        return leaves if leaves else [Node(is_leaf=True)]

    def _build_internal_from_children(self, children: list[Node]) -> Node:
        """
        Build internal node(s) from a list of children using rolling hash for splits.

        Args:
            children: List of Node objects

        Returns:
            Node (single child, or newly created internal node)
        """
        if len(children) == 0:
            raise ValueError("Cannot build internal node with no children")

        if len(children) == 1:
            # Single child - just return it (no need for parent)
            return children[0]

        # Build internal nodes using rolling hash to determine split points
        # Strategy: Don't split unless we have at least 2 children on BOTH sides
        internal_nodes = []
        current_internal = Node(is_leaf=False)
        roll_hash = self.tree.seed  # Start with seed

        for i, child in enumerate(children):
            # Store or reuse child hash
            if hasattr(child, '_reused_hash'):
                child_hash = getattr(child, '_reused_hash')
                delattr(child, '_reused_hash')
            else:
                child_hash = self.tree._store_node(child)

            current_internal.values.append(child_hash)

            # Update rolling hash with the child hash
            hash_bytes = str(child_hash).encode('utf-8')
            roll_hash = self.tree._rolling_hash(roll_hash, hash_bytes)

            # Add separator key (first key of next child)
            if i < len(children) - 1:
                next_child = children[i + 1]
                # Get the first actual key from the next child's subtree
                separator = self.tree._get_first_key(next_child)
                if separator is not None:
                    current_internal.keys.append(separator)

                    # Update rolling hash with separator key
                    sep_bytes = str(separator).encode('utf-8')
                    roll_hash = self.tree._rolling_hash(roll_hash, sep_bytes)

                    # Check if we should split here using rolling hash
                    # Require:
                    # - At least 2 children in current node
                    # - At least 2 children remaining (including next)
                    MIN_CHILDREN = 2
                    children_remaining = len(children) - i - 1
                    if (roll_hash < self.tree.pattern and
                        len(current_internal.values) >= MIN_CHILDREN and
                        children_remaining >= MIN_CHILDREN):
                        # Split point! Validate and save current internal
                        if self.tree.validate:
                            current_internal.validate(self.tree.store, context="_build_internal_from_children (split)")
                        internal_nodes.append(current_internal)
                        current_internal = Node(is_leaf=False)
                        roll_hash = self.tree.seed  # Reset hash for next node

        # Add the last internal node (but only if it has multiple children)
        if current_internal.values:
            if len(current_internal.values) == 1 and not internal_nodes:
                # Only one child total - just return it directly
                child_hash = current_internal.values[0]
                child = self.tree._get_node(child_hash)
                if child is None:
                    raise ValueError(f"Child node {child_hash} not found in store")
                return child
            elif len(current_internal.values) > 1:
                # Validate before adding
                if self.tree.validate:
                    current_internal.validate(self.tree.store, context="_build_internal_from_children (end)")
                internal_nodes.append(current_internal)
            elif internal_nodes:
                # Single child but we already have other nodes - validate and add it
                if self.tree.validate:
                    current_internal.validate(self.tree.store, context="_build_internal_from_children (single child)")
                internal_nodes.append(current_internal)

        # Handle edge cases
        if len(internal_nodes) == 0:
            raise ValueError("No internal nodes created")
        elif len(internal_nodes) == 1:
            # Single internal node
            node = internal_nodes[0]
            if len(node.values) == 1:
                # Unwrap single-child internal node - return the child directly
                child_hash = node.values[0]
                child = self.tree._get_node(child_hash)
                if child is None:
                    raise ValueError(f"Child node {child_hash} not found in store")
                return child
            elif len(node.values) == 0:
                raise ValueError("Internal node has no children")
            else:
                # Validate before returning
                if self.tree.validate:
                    node.validate(self.tree.store, context="_build_internal_from_children (return single)")
                return node
        else:
            # Multiple internal nodes - build parent recursively
            return self._build_internal_from_children(internal_nodes)
