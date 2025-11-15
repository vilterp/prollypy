# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Diff algorithm for ProllyTree.

Efficiently computes differences between two trees by skipping identical subtrees
based on content hashes.
"""

from dataclasses import dataclass
from typing import Any, Iterator, Union, Optional
from store import Store
from cursor import TreeCursor


@dataclass(frozen=True)
class Added:
    """A key-value pair was added."""
    key: Any
    value: Any

    def __repr__(self):
        return f"Added({self.key!r}, {self.value!r})"


@dataclass(frozen=True)
class Deleted:
    """A key was deleted."""
    key: Any
    old_value: Any

    def __repr__(self):
        return f"Deleted({self.key!r}, {self.old_value!r})"


@dataclass(frozen=True)
class Modified:
    """A key's value was modified."""
    key: Any
    old_value: Any
    new_value: Any

    def __repr__(self):
        return f"Modified({self.key!r}, {self.old_value!r} -> {self.new_value!r})"


DiffEvent = Union[Added, Deleted, Modified]


@dataclass
class DiffStats:
    """Statistics from a diff operation."""
    subtrees_skipped: int = 0
    nodes_compared: int = 0

    def __repr__(self):
        return f"DiffStats(subtrees_skipped={self.subtrees_skipped}, nodes_compared={self.nodes_compared})"


class Differ:
    """
    Diff two ProllyTree structures with statistics tracking.
    """

    def __init__(self, store: Store):
        """
        Initialize differ with store.

        Args:
            store: Storage backend containing both trees
        """
        self.store = store
        self.stats = DiffStats()

    def diff(self, old_hash: str, new_hash: str, prefix: str = None) -> Iterator[DiffEvent]:
        """
        Compute differences between two trees using cursor-based traversal.

        This algorithm handles trees with different structures by comparing
        key-value pairs directly, regardless of tree shape.

        When a prefix is provided, uses O(log n) seeking to jump directly to
        the prefix location in both trees, avoiding unnecessary iteration.

        Args:
            old_hash: Root hash of the old tree
            new_hash: Root hash of the new tree
            prefix: Optional key prefix to filter diff results

        Yields:
            DiffEvent objects (Added, Deleted, or Modified) in key order
        """
        # Reset stats for new diff operation
        self.stats = DiffStats()
        self.prefix = prefix

        # If hashes are the same, trees are identical - no diff needed
        if old_hash == new_hash:
            self.stats.subtrees_skipped += 1
            return

        # Create cursors for both trees, seeking to prefix if provided
        # This provides O(log n) performance when diffing with a prefix filter
        old_cursor = TreeCursor(self.store, old_hash, seek_to=prefix)
        new_cursor = TreeCursor(self.store, new_hash, seek_to=prefix)

        # Get first entries
        old_entry = old_cursor.next()
        new_entry = new_cursor.next()

        # Track if we've found any matches (for early termination with prefix)
        found_match = False

        # Merge-like traversal of both trees
        while old_entry is not None or new_entry is not None:
            # Early termination: if we have a prefix and both entries don't match,
            # and we've already found matches, we're past the prefix range
            if prefix:
                old_matches = old_entry is not None and self._matches_prefix(old_entry[0])
                new_matches = new_entry is not None and self._matches_prefix(new_entry[0])

                if found_match and not old_matches and not new_matches:
                    # We've passed the prefix range - stop
                    break

            # Check for subtree skipping opportunity
            if old_entry is not None and new_entry is not None:
                old_next_hash = old_cursor.peek_next_hash()
                new_next_hash = new_cursor.peek_next_hash()

                if old_next_hash and new_next_hash and old_next_hash == new_next_hash:
                    # Same subtree coming up - skip it!
                    self.stats.subtrees_skipped += 1
                    old_cursor.skip_subtree(old_next_hash)
                    new_cursor.skip_subtree(new_next_hash)
                    old_entry = old_cursor.next()
                    new_entry = new_cursor.next()
                    continue

            if old_entry is None:
                # Only new entries remain - all additions
                while new_entry is not None:
                    if self._matches_prefix(new_entry[0]):
                        found_match = True
                        yield Added(new_entry[0], new_entry[1])
                    elif prefix and found_match:
                        # Past prefix range
                        break
                    new_entry = new_cursor.next()
                break

            if new_entry is None:
                # Only old entries remain - all deletions
                while old_entry is not None:
                    if self._matches_prefix(old_entry[0]):
                        found_match = True
                        yield Deleted(old_entry[0], old_entry[1])
                    elif prefix and found_match:
                        # Past prefix range
                        break
                    old_entry = old_cursor.next()
                break

            # Both have entries - compare keys
            old_key, old_value = old_entry
            new_key, new_value = new_entry

            if old_key < new_key:
                # Key only in old tree - deleted
                if self._matches_prefix(old_key):
                    found_match = True
                    yield Deleted(old_key, old_value)
                old_entry = old_cursor.next()
            elif old_key > new_key:
                # Key only in new tree - added
                if self._matches_prefix(new_key):
                    found_match = True
                    yield Added(new_key, new_value)
                new_entry = new_cursor.next()
            else:
                # Same key in both trees
                if old_value != new_value:
                    # Value changed - modified
                    if self._matches_prefix(old_key):
                        found_match = True
                        yield Modified(old_key, old_value, new_value)
                # else: values are identical, no diff event needed

                # Advance both cursors
                old_entry = old_cursor.next()
                new_entry = new_cursor.next()

    def get_stats(self) -> DiffStats:
        """Get statistics from the most recent diff operation."""
        return self.stats

    def _matches_prefix(self, key: Any) -> bool:
        """Check if a key matches the prefix filter."""
        if self.prefix is None:
            return True
        # Convert key to string for comparison
        key_str = str(key)
        return key_str.startswith(self.prefix)

    def _diff_nodes(self, old_node, new_node) -> Iterator[DiffEvent]:
        """
        Recursively diff two nodes.

        Args:
            old_node: Old tree node
            new_node: New tree node

        Yields:
            DiffEvent objects in key order
        """
        if old_node.is_leaf and new_node.is_leaf:
            # Both are leaves - compare entries directly
            yield from self._diff_leaves(old_node, new_node)
        elif old_node.is_leaf and not new_node.is_leaf:
            # Old is leaf, new is internal - handle mixed case
            # Get all entries from old leaf
            old_entries = dict(zip(old_node.keys, old_node.values))

            # Traverse new internal node and compare
            yield from self._diff_leaf_vs_internal(old_entries, new_node)
        elif not old_node.is_leaf and new_node.is_leaf:
            # Old is internal, new is leaf - handle mixed case
            # Get all entries from new leaf
            new_entries = dict(zip(new_node.keys, new_node.values))

            # Traverse old internal node and compare
            yield from self._diff_internal_vs_leaf(old_node, new_entries)
        else:
            # Both are internal nodes - traverse in parallel
            yield from self._diff_internal_nodes(old_node, new_node)

    def _diff_leaves(self, old_node, new_node) -> Iterator[DiffEvent]:
        """
        Diff two leaf nodes.

        Args:
            old_node: Old leaf node
            new_node: New leaf node

        Yields:
            DiffEvent objects in key order
        """
        old_entries = dict(zip(old_node.keys, old_node.values))
        new_entries = dict(zip(new_node.keys, new_node.values))

        # All keys from both leaves
        all_keys = sorted(set(old_entries.keys()) | set(new_entries.keys()))

        for key in all_keys:
            # Skip keys that don't match prefix
            if not self._matches_prefix(key):
                continue

            old_has = key in old_entries
            new_has = key in new_entries

            if old_has and new_has:
                # Key exists in both
                if old_entries[key] != new_entries[key]:
                    yield Modified(key, old_entries[key], new_entries[key])
            elif new_has:
                # Only in new
                yield Added(key, new_entries[key])
            else:
                # Only in old
                yield Deleted(key, old_entries[key])

    def _diff_internal_nodes(self, old_node, new_node) -> Iterator[DiffEvent]:
        """
        Diff two internal nodes by traversing children in parallel.

        Args:
            old_node: Old internal node
            new_node: New internal node

        Yields:
            DiffEvent objects in key order
        """
        # Build child ranges for both nodes
        old_children = self._get_child_ranges(old_node)
        new_children = self._get_child_ranges(new_node)

        old_idx = 0
        new_idx = 0

        while old_idx < len(old_children) or new_idx < len(new_children):
            if old_idx >= len(old_children):
                # No more old children - all remaining new children are additions
                new_hash, _, _ = new_children[new_idx]
                new_child = self.store.get_node(new_hash)
                yield from self._yield_all_additions(new_child)
                new_idx += 1
            elif new_idx >= len(new_children):
                # No more new children - all remaining old children are deletions
                old_hash, _, _ = old_children[old_idx]
                old_child = self.store.get_node(old_hash)
                yield from self._yield_all_deletions(old_child)
                old_idx += 1
            else:
                old_hash, old_lower, old_upper = old_children[old_idx]
                new_hash, new_lower, new_upper = new_children[new_idx]

                # Check if ranges overlap
                if old_upper is not None and new_lower is not None and old_upper <= new_lower:
                    # Old range is entirely before new range - deletions
                    old_child = self.store.get_node(old_hash)
                    yield from self._yield_all_deletions(old_child)
                    old_idx += 1
                elif new_upper is not None and old_lower is not None and new_upper <= old_lower:
                    # New range is entirely before old range - additions
                    new_child = self.store.get_node(new_hash)
                    yield from self._yield_all_additions(new_child)
                    new_idx += 1
                else:
                    # Ranges overlap - need to diff these children
                    if old_hash == new_hash:
                        # Identical subtree - skip it!
                        self.stats.subtrees_skipped += 1
                        old_idx += 1
                        new_idx += 1
                    else:
                        # Different subtrees - recurse
                        old_child = self.store.get_node(old_hash)
                        new_child = self.store.get_node(new_hash)
                        self.stats.nodes_compared += 1
                        yield from self._diff_nodes(old_child, new_child)
                        old_idx += 1
                        new_idx += 1

    def _get_child_ranges(self, node):
        """
        Get list of (child_hash, lower_bound, upper_bound) for each child in an internal node.

        Args:
            node: Internal node

        Returns:
            List of tuples (child_hash, lower_bound, upper_bound)
        """
        children = []
        for i, child_hash in enumerate(node.values):
            lower_bound = node.keys[i - 1] if i > 0 else None
            upper_bound = node.keys[i] if i < len(node.keys) else None
            children.append((child_hash, lower_bound, upper_bound))
        return children

    def _diff_leaf_vs_internal(self, old_entries: dict, new_node) -> Iterator[DiffEvent]:
        """
        Diff a leaf node against an internal node.

        Args:
            old_entries: Dictionary of old leaf entries
            new_node: New internal node

        Yields:
            DiffEvent objects in key order
        """
        # Collect all entries from new internal node
        new_entries = {}
        for key, value in self._collect_all_entries(new_node):
            new_entries[key] = value

        # Compare
        all_keys = sorted(set(old_entries.keys()) | set(new_entries.keys()))
        for key in all_keys:
            # Skip keys that don't match prefix
            if not self._matches_prefix(key):
                continue

            old_has = key in old_entries
            new_has = key in new_entries

            if old_has and new_has:
                if old_entries[key] != new_entries[key]:
                    yield Modified(key, old_entries[key], new_entries[key])
            elif new_has:
                yield Added(key, new_entries[key])
            else:
                yield Deleted(key, old_entries[key])

    def _diff_internal_vs_leaf(self, old_node, new_entries: dict) -> Iterator[DiffEvent]:
        """
        Diff an internal node against a leaf node.

        Args:
            old_node: Old internal node
            new_entries: Dictionary of new leaf entries

        Yields:
            DiffEvent objects in key order
        """
        # Collect all entries from old internal node
        old_entries = {}
        for key, value in self._collect_all_entries(old_node):
            old_entries[key] = value

        # Compare
        all_keys = sorted(set(old_entries.keys()) | set(new_entries.keys()))
        for key in all_keys:
            # Skip keys that don't match prefix
            if not self._matches_prefix(key):
                continue

            old_has = key in old_entries
            new_has = key in new_entries

            if old_has and new_has:
                if old_entries[key] != new_entries[key]:
                    yield Modified(key, old_entries[key], new_entries[key])
            elif new_has:
                yield Added(key, new_entries[key])
            else:
                yield Deleted(key, old_entries[key])

    def _collect_all_entries(self, node) -> Iterator[tuple]:
        """
        Recursively collect all entries from a node.

        Args:
            node: Node to collect from

        Yields:
            Tuples of (key, value)
        """
        if node.is_leaf:
            for key, value in zip(node.keys, node.values):
                yield (key, value)
        else:
            for child_hash in node.values:
                child = self.store.get_node(child_hash)
                yield from self._collect_all_entries(child)

    def _yield_all_additions(self, node) -> Iterator[Added]:
        """
        Yield Added events for all entries in a node.

        Args:
            node: Node containing additions

        Yields:
            Added events for all entries
        """
        for key, value in self._collect_all_entries(node):
            if self._matches_prefix(key):
                yield Added(key, value)

    def _yield_all_deletions(self, node) -> Iterator[Deleted]:
        """
        Yield Deleted events for all entries in a node.

        Args:
            node: Node containing deletions

        Yields:
            Deleted events for all entries
        """
        for key, value in self._collect_all_entries(node):
            if self._matches_prefix(key):
                yield Deleted(key, value)


# Backward compatibility function
def diff(store: Store, old_hash: str, new_hash: str, prefix: str = None) -> Iterator[DiffEvent]:
    """
    Compute differences between two trees (backward compatibility wrapper).

    Args:
        store: Storage backend containing both trees
        old_hash: Root hash of the old tree
        new_hash: Root hash of the new tree
        prefix: Optional key prefix to filter diff results

    Yields:
        DiffEvent objects (Added, Deleted, or Modified) in key order
    """
    differ = Differ(store)
    yield from differ.diff(old_hash, new_hash, prefix=prefix)
