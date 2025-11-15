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


class LevelBuilder:
    """
    Builder for a single level in the tree during incremental construction.

    Each level accumulates children (entries for leaves, nodes for internal levels)
    and emits completed nodes when they reach split points based on rolling hash.
    """

    def __init__(self, tree: 'ProllyTree', is_leaf: bool):
        self.tree = tree
        self.is_leaf = is_leaf
        self.current_keys = []
        self.current_values = []
        self.roll_hash = tree.seed
        self.MIN_SIZE = 2

    def add_entry(self, key: str, value: str) -> Optional[Node]:
        """Add a key-value entry to current leaf. Returns completed node if split."""
        assert self.is_leaf, "add_entry only for leaf levels"

        self.current_keys.append(key)
        self.current_values.append(value)

        # Update rolling hash
        key_bytes = str(key).encode('utf-8')
        value_bytes = str(value).encode('utf-8')
        self.roll_hash = self.tree._rolling_hash(self.roll_hash, key_bytes)
        self.roll_hash = self.tree._rolling_hash(self.roll_hash, value_bytes)

        # Check for split
        if len(self.current_keys) >= self.MIN_SIZE and self.roll_hash < self.tree.pattern:
            return self._emit_current()

        return None

    def add_child(self, child: Node) -> Optional[Node]:
        """Add a child node to current internal. Returns completed node if split."""
        assert not self.is_leaf, "add_child only for internal levels"

        # Add separator for previous child (if any)
        if len(self.current_values) > 0:
            # Separator is the first key of THIS child (which comes after previous child)
            separator = self.tree._get_first_key(child)
            if separator:
                self.current_keys.append(separator)

                # Update rolling hash with separator
                sep_bytes = str(separator).encode('utf-8')
                self.roll_hash = self.tree._rolling_hash(self.roll_hash, sep_bytes)

        # Get or compute child hash
        if hasattr(child, '_reused_hash') and child._reused_hash:
            child_hash = child._reused_hash
            delattr(child, '_reused_hash')
        else:
            child_hash = self.tree._store_node(child)

        self.current_values.append(child_hash)

        # Update rolling hash with child hash
        hash_bytes = str(child_hash).encode('utf-8')
        self.roll_hash = self.tree._rolling_hash(self.roll_hash, hash_bytes)

        # Check for split (need at least 2 children on both sides)
        if (len(self.current_values) >= self.MIN_SIZE and
            self.roll_hash < self.tree.pattern):
            return self._emit_current()

        return None

    def finalize(self) -> Optional[Node]:
        """Emit final node with remaining data."""
        if not self.current_keys and not self.current_values:
            return None
        return self._emit_current()

    def _emit_current(self) -> Optional[Node]:
        """Create node from current accumulated data and reset."""
        if not self.current_values:
            return None

        node = Node(is_leaf=self.is_leaf)
        node.keys = self.current_keys
        node.values = self.current_values

        if self.tree.validate:
            context = "LevelBuilder (leaf)" if self.is_leaf else "LevelBuilder (internal)"
            node.validate(self.tree.store, context=context)

        # Reset for next node
        self.current_keys = []
        self.current_values = []
        self.roll_hash = self.tree.seed

        return node


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

        # Stack for streaming tree construction
        # stack[0] = leaf level, stack[1] = first internal level, etc.
        self.stack: list[LevelBuilder] = [LevelBuilder(tree, is_leaf=True)]

    def rebuild(self) -> Node:
        """
        Rebuild tree with mutations using stack-based streaming merge.

        Processes the merge stream incrementally, building the tree bottom-up
        using a stack of level builders. Never materializes full child lists.

        Returns:
            New root node with mutations applied
        """
        # Process merge stream, adding items to the stack
        for item in self._merge_stream():
            if isinstance(item, SubtreeRef):
                # Reused subtree - fetch and add as child
                child_node = self.tree._get_node(item.hash)
                if child_node is None:
                    raise ValueError(f"Referenced subtree {item.hash} not found in store")
                child_node._reused_hash = item.hash
                self._add_child_to_stack(child_node)
            else:
                # Regular entry - add to leaf level
                key, value = item
                self._add_entry_to_stack(key, value)

        # Finalize all levels from bottom to top
        return self._finalize_stack()

    def _add_entry_to_stack(self, key: str, value: str):
        """Add a key-value entry to the leaf level, propagating splits up the stack."""
        completed = self.stack[0].add_entry(key, value)
        if completed:
            self._add_child_to_stack(completed)

    def _add_child_to_stack(self, child: Node):
        """Add a child node to the parent level, creating levels as needed."""
        level = 1
        current_child = child

        while current_child is not None:
            # Ensure parent level exists
            if level >= len(self.stack):
                self.stack.append(LevelBuilder(self.tree, is_leaf=False))

            # Add child to parent level
            completed = self.stack[level].add_child(current_child)
            if completed is None:
                break  # No split, we're done

            # Parent level split, propagate upward
            current_child = completed
            level += 1

    def _finalize_stack(self) -> Node:
        """Finalize all levels and return final root node."""
        # Finalize from bottom to top, propagating final nodes upward
        for level in range(len(self.stack)):
            completed = self.stack[level].finalize()
            if completed is None:
                continue

            # Propagate final node upward
            if level + 1 < len(self.stack):
                # Add to existing parent level
                parent_completed = self.stack[level + 1].add_child(completed)
                # If parent split during this add, we need to handle it
                if parent_completed is not None:
                    # Parent split - need to propagate even further up
                    current_level = level + 2
                    current_child = parent_completed
                    while current_child is not None:
                        if current_level >= len(self.stack):
                            self.stack.append(LevelBuilder(self.tree, is_leaf=False))
                        next_completed = self.stack[current_level].add_child(current_child)
                        current_child = next_completed
                        current_level += 1
            else:
                # Need new parent level for the final node
                parent = LevelBuilder(self.tree, is_leaf=False)
                parent.add_child(completed)
                self.stack.append(parent)

        # The top level should have exactly one node (the root)
        # Finalize it to get the final root
        if len(self.stack) > 0:
            root = self.stack[-1].finalize()
            if root is not None:
                return root

        # If we get here, return an empty tree
        return Node(is_leaf=True)

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
