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
ProllyTree implementation with content-based splitting.

Features:
- Content-addressed nodes (hash of contents)
- Rolling hash-based splitting (Rabin fingerprinting)
- Incremental batch insert with subtree reuse
- Pluggable storage backends via Store protocol
"""

import hashlib
from typing import Optional, Iterator
from .node import Node
from .store import Store, MemoryStore, CachedFSStore
from .cursor import TreeCursor


class BatchStats:
    """Statistics for a single batch operation."""
    __slots__ = ('nodes_created', 'leaves_created', 'internals_created',
                 'nodes_reused', 'subtrees_reused', 'nodes_read')

    def __init__(self):
        self.nodes_created: int = 0
        self.leaves_created: int = 0
        self.internals_created: int = 0
        self.nodes_reused: int = 0
        self.subtrees_reused: int = 0
        self.nodes_read: int = 0

    def reset(self):
        """Reset all counters to zero."""
        self.nodes_created = 0
        self.leaves_created = 0
        self.internals_created = 0
        self.nodes_reused = 0
        self.subtrees_reused = 0
        self.nodes_read = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for backwards compatibility."""
        return {
            'nodes_created': self.nodes_created,
            'leaves_created': self.leaves_created,
            'internals_created': self.internals_created,
            'nodes_reused': self.nodes_reused,
            'subtrees_reused': self.subtrees_reused,
            'nodes_read': self.nodes_read,
        }


class ProllyTree:
    def __init__(self, pattern=0.25, seed=42, store: Optional[Store] = None, validate=False):
        """
        Initialize ProllyTree with content-based splitting.

        Args:
            pattern: Split probability (0.0 to 1.0). Lower = larger nodes.
                    Default 0.25 means ~4 entries per node on average.
            seed: Seed for rolling hash function for reproducibility
            store: Storage backend (defaults to MemoryStore if not provided)
            validate: If True, validate node structure during tree building (slower)
        """
        self.pattern = int(pattern * (2**32))  # Convert to uint32 threshold
        self.seed = seed
        self.store = store if store is not None else MemoryStore()
        self.validate = validate

        self.root = Node(is_leaf=True)

        # Operation statistics
        self.stats = BatchStats()

    def reset_stats(self):
        """Reset operation statistics for a new batch"""
        self.stats.reset()

    def _rolling_hash(self, current_hash: int, data: bytes) -> int:
        """
        Deterministic rolling hash using hashlib (SHA256).
        Updates the hash with new data.

        Args:
            current_hash: Current hash value (use self.seed for initial)
            data: bytes-like object to add to the hash

        Returns:
            Updated hash value (uint32)
        """
        # Use SHA256 for deterministic hashing - fast C implementation
        # Combine current hash with new data
        combined = current_hash.to_bytes(4, byteorder='big') + data
        hash_bytes = hashlib.sha256(combined).digest()
        # Take first 4 bytes and convert to uint32
        return int.from_bytes(hash_bytes[:4], byteorder='big')

    def _hash_node(self, node: Node) -> str:
        """
        Compute content hash for a node.

        For leaf nodes: hash the key-value pairs
        For internal nodes: hash the (separator_key, child_hash) pairs
        """
        content = []
        if node.is_leaf:
            # Hash key-value pairs
            for key, value in zip(node.keys, node.values):
                content.append(f"{key}:{value}".encode('utf-8'))
        else:
            # Hash separator keys and child hashes
            for i, child_hash in enumerate(node.values):
                if i < len(node.keys):
                    content.append(f"{node.keys[i]}:{child_hash}".encode('utf-8'))
                else:
                    content.append(f"_:{child_hash}".encode('utf-8'))

        # Combine all content and hash
        combined = b'|'.join(content)
        return hashlib.sha256(combined).hexdigest()[:16]  # Use first 16 chars for readability

    def _store_node(self, node: Node) -> str:
        """Store node and return its content-based hash"""
        node_hash = self._hash_node(node)

        # Only store if not already present (deduplication)
        existing = self.store.get_node(node_hash)
        if existing is None:
            self.store.put_node(node_hash, node)
            self.stats.nodes_created += 1
            if node.is_leaf:
                self.stats.leaves_created += 1
            else:
                self.stats.internals_created += 1
        else:
            self.stats.nodes_reused += 1

        return node_hash

    def _get_node(self, node_hash: str) -> Optional[Node]:
        """Retrieve node by hash"""
        return self.store.get_node(node_hash)

    def insert_batch(self, mutations: list[tuple[str, str]], verbose: bool = True) -> dict[str, int]:
        """
        Incrementally insert a batch of (key, value) pairs.
        mutations: sorted list of (key, value) tuples
        Returns: dict with operation stats
        """
        import time

        # Track timing
        start_time = time.time()

        # Reset stats for this batch
        self.reset_stats()

        # Track cache stats before this batch (if using CachedFSStore)
        cache_stats_before = None
        if isinstance(self.store, CachedFSStore):
            cache_stats_before = {
                'hits': self.store.cache_hits,
                'misses': self.store.cache_misses
            }

        # Rebuild tree with mutations (always quiet during rebuild)
        new_root = self._rebuild_with_mutations(self.root, mutations, verbose=False)

        # Store the new root (unless it was reused)
        if new_root is not self.root:
            self._store_node(new_root)

        self.root = new_root

        # Validate if enabled
        if self.validate:
            is_valid, error_msg, position, prev_key, current_key = self.validate_sorted()
            if not is_valid:
                # Print some debug info
                print(f"\n!!! VALIDATION FAILED !!!")
                print(f"  Position: {position}")
                print(f"  Previous key: {prev_key}")
                print(f"  Current key: {current_key}")
                print(f"  Batch size: {len(mutations)}")
                if mutations:
                    print(f"  First mutation key: {mutations[0][0]}")
                    print(f"  Last mutation key: {mutations[-1][0]}")
                raise ValueError(f"Tree validation failed after batch insert: {error_msg} - {prev_key} > {current_key}")

        stats = self._summarize_ops()

        # Calculate timing
        elapsed = time.time() - start_time
        rows_per_sec = len(mutations) / elapsed if elapsed > 0 else 0

        # Print single-line batch summary
        if verbose:
            summary_parts = [f"Inserted {len(mutations)} rows"]
            summary_parts.append(f"{stats['nodes_created']} new nodes created")
            summary_parts.append(f"{rows_per_sec:,.0f} rows/sec")

            # Add cache stats if using CachedFSStore
            if isinstance(self.store, CachedFSStore) and cache_stats_before:
                hits_delta = self.store.cache_hits - cache_stats_before['hits']
                misses_delta = self.store.cache_misses - cache_stats_before['misses']
                summary_parts.append(f"{hits_delta} cache hits")
                summary_parts.append(f"{misses_delta} cache misses")

                # Add average node sizes from CachedFSStore
                size_stats = self.store.get_size_stats()
                if size_stats['avg_leaf_size'] > 0:
                    summary_parts.append(f"avg leaf: {size_stats['avg_leaf_size']:.0f}B")
                if size_stats['avg_internal_size'] > 0:
                    summary_parts.append(f"avg internal: {size_stats['avg_internal_size']:.0f}B")

            print("; ".join(summary_parts))

        return stats

    def _rebuild_with_mutations(self, node: Node, mutations: list[tuple[str, str]], verbose: bool = True) -> Node:
        """
        Core incremental rebuild logic.
        Returns: new node (possibly with different structure)
        """
        if verbose:
            node_type = 'Leaf' if node.is_leaf else 'Internal'
            print(f"\n_rebuild_with_mutations: {node_type} node with {len(node.keys)} keys, {len(mutations)} mutations")

        if not mutations:
            # No mutations for this subtree - REUSE it!
            if verbose:
                print(f"  -> No mutations, reusing node")
            return node

        if node.is_leaf:
            # Leaf node: merge old data with mutations
            if verbose:
                print(f"  -> Leaf node, merging {len(node.keys)} existing + {len(mutations)} new entries...")
            merged = self._merge_sorted(
                list(zip(node.keys, node.values)),
                mutations
            )
            if verbose:
                print(f"  -> Merged to {len(merged)} total entries")

            # Build new leaf nodes (may split if too large)
            new_leaves = self._build_leaves(merged)
            if verbose:
                print(f"  -> Built {len(new_leaves)} leaf nodes")

            if len(new_leaves) == 1:
                return new_leaves[0]
            else:
                # Multiple leaves - need parent (may recursively split if too many)
                return self._build_internal_from_children(new_leaves, verbose)

        else:
            # Internal node: partition mutations to children and rebuild recursively
            if verbose:
                print(f"  -> Internal node, partitioning {len(mutations)} mutations to children")

            # Recursively rebuild children that have mutations
            new_children = []
            mut_idx = 0

            for i, child_hash in enumerate(node.values):
                # Find mutations for this child
                # For child i, mutations go to it if:
                # - i == 0: key < separator[0]
                # - 0 < i < len(node.keys): separator[i-1] <= key < separator[i]
                # - i == len(node.keys): key >= separator[-1]
                child_mutations = []

                while mut_idx < len(mutations):
                    key = mutations[mut_idx][0]

                    # Determine if this mutation belongs to this child
                    if i == 0:
                        # First child: all keys < first separator
                        if len(node.keys) == 0 or key < node.keys[0]:
                            child_mutations.append(mutations[mut_idx])
                            mut_idx += 1
                        else:
                            break
                    elif i < len(node.keys):
                        # Middle child: separator[i-1] <= key < separator[i]
                        if key < node.keys[i]:
                            child_mutations.append(mutations[mut_idx])
                            mut_idx += 1
                        else:
                            break
                    else:
                        # Last child: all remaining keys
                        child_mutations.append(mutations[mut_idx])
                        mut_idx += 1

                # Rebuild child (or reuse if no mutations)
                child_node = self._get_node(child_hash)
                if child_node is None:
                    raise ValueError(f"Child node {child_hash} not found in store")
                new_child = self._rebuild_with_mutations(child_node, child_mutations, verbose)

                # The rebuild might return multiple nodes (if split), or a single node
                if isinstance(new_child, list):
                    new_children.extend(new_child)
                else:
                    new_children.append(new_child)

            if verbose:
                print(f"  -> Rebuilt {len(new_children)} children from {len(node.values)} original children")

            # Now rebuild internal structure from new children
            if len(new_children) == 1:
                # Single child - unwrap it
                return new_children[0]
            else:
                # Build new internal node(s) from children
                new_node = self._build_internal_from_children(new_children, verbose)
                if self.validate:
                    new_node.validate(self.store, context="_rebuild_with_mutations (internal node rebuild)")
                return new_node

    def _get_first_key(self, node: Node) -> Optional[str]:
        """
        Get the first actual key in a node's subtree.

        For leaf nodes, returns keys[0].
        For internal nodes, recursively descends to leftmost leaf.

        Args:
            node: Node to get first key from

        Returns:
            First key in subtree, or None if node has no keys
        """
        if node.is_leaf:
            return node.keys[0] if len(node.keys) > 0 else None
        else:
            # Internal node - descend to leftmost child
            if len(node.values) == 0:
                return None
            child_hash = node.values[0]
            child = self._get_node(child_hash)
            if child is None:
                return None
            return self._get_first_key(child)

    def _build_internal_from_children(self, children: list[Node], verbose: bool = False) -> Node:
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
        roll_hash = self.seed  # Start with seed

        for i, child in enumerate(children):
            # Store or reuse child hash
            if hasattr(child, '_reused_hash'):
                child_hash = getattr(child, '_reused_hash')
                delattr(child, '_reused_hash')
            else:
                child_hash = self._store_node(child)

            current_internal.values.append(child_hash)

            # Update rolling hash with the child hash
            hash_bytes = str(child_hash).encode('utf-8')
            roll_hash = self._rolling_hash(roll_hash, hash_bytes)

            # Add separator key (first key of next child)
            if i < len(children) - 1:
                next_child = children[i + 1]
                # Get the first actual key from the next child's subtree
                separator = self._get_first_key(next_child)
                if separator is not None:
                    current_internal.keys.append(separator)

                    # Update rolling hash with separator key
                    sep_bytes = str(separator).encode('utf-8')
                    roll_hash = self._rolling_hash(roll_hash, sep_bytes)

                    # Check if we should split here using rolling hash
                    # Require:
                    # - At least 2 children in current node
                    # - At least 2 children remaining (including next)
                    MIN_CHILDREN = 2
                    children_remaining = len(children) - i - 1
                    if (roll_hash < self.pattern and
                        len(current_internal.values) >= MIN_CHILDREN and
                        children_remaining >= MIN_CHILDREN):
                        # Split point! Validate and save current internal
                        if self.validate:
                            current_internal.validate(self.store, context="_build_internal_from_children (split)")
                        internal_nodes.append(current_internal)
                        current_internal = Node(is_leaf=False)
                        roll_hash = self.seed  # Reset hash for next node
                        if verbose:
                            print(f"  -> Internal node split at separator {separator} (hash={roll_hash} < {self.pattern})")
                else:
                    # Empty child node - skip it
                    if verbose:
                        print(f"  -> Warning: child {i+1} has no keys, skipping separator")

        # Add the last internal node (but only if it has multiple children)
        if current_internal.values:
            if len(current_internal.values) == 1 and not internal_nodes:
                # Only one child total - just return it directly
                child_hash = current_internal.values[0]
                child = self._get_node(child_hash)
                if child is None:
                    raise ValueError(f"Child node {child_hash} not found in store")
                return child
            elif len(current_internal.values) > 1:
                # Validate before adding
                if self.validate:
                    current_internal.validate(self.store, context="_build_internal_from_children (end)")
                internal_nodes.append(current_internal)
            elif internal_nodes:
                # Single child but we already have other nodes - validate and add it
                if self.validate:
                    current_internal.validate(self.store, context="_build_internal_from_children (single child)")
                internal_nodes.append(current_internal)

        # Handle edge cases
        if len(internal_nodes) == 0:
            raise ValueError("No internal nodes created")
        elif len(internal_nodes) == 1:
            # Single internal node
            node = internal_nodes[0]
            if len(node.values) == 1:
                # Unwrap single-child internal node - return the child directly
                if verbose:
                    print(f"  -> Unwrapping single-child internal node")
                child_hash = node.values[0]
                child = self._get_node(child_hash)
                if child is None:
                    raise ValueError(f"Child node {child_hash} not found in store")
                return child
            elif len(node.values) == 0:
                raise ValueError("Internal node has no children")
            else:
                # Validate before returning
                if self.validate:
                    node.validate(self.store, context="_build_internal_from_children (return single)")
                return node
        else:
            # Multiple internal nodes - build parent recursively
            if verbose:
                print(f"  -> Created {len(internal_nodes)} internal nodes, building parent...")
            return self._build_internal_from_children(internal_nodes, verbose)

    def _merge_sorted(self, old_items: list[tuple[str, str]], new_items: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge two sorted lists of (key, value) tuples"""
        result = []
        i, j = 0, 0

        while i < len(old_items) and j < len(new_items):
            if old_items[i][0] < new_items[j][0]:
                result.append(old_items[i])
                i += 1
            elif old_items[i][0] == new_items[j][0]:
                # New value overwrites old
                result.append(new_items[j])
                i += 1
                j += 1
            else:
                result.append(new_items[j])
                j += 1

        result.extend(old_items[i:])
        result.extend(new_items[j:])
        return result

    def _build_leaves(self, items: list[tuple[str, str]]) -> list[Node]:
        """
        Build leaf nodes from sorted items using rolling hash for splits.

        Split points are determined by rolling hash being below pattern threshold,
        with a minimum of 2 entries per node to avoid degenerate splits.
        """
        if not items:
            return []

        MIN_NODE_SIZE = 2  # Minimum entries per node to avoid degenerate trees

        leaves = []
        current_keys = []
        current_values = []
        roll_hash = self.seed  # Start with seed

        for i, (key, value) in enumerate(items):
            current_keys.append(key)
            current_values.append(value)

            # Update rolling hash with the key and value bytes
            key_bytes = str(key).encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            roll_hash = self._rolling_hash(roll_hash, key_bytes)
            roll_hash = self._rolling_hash(roll_hash, value_bytes)

            # Split if: (1) have minimum entries AND hash below pattern OR (2) last item
            has_min = len(current_keys) >= MIN_NODE_SIZE
            should_split = (has_min and roll_hash < self.pattern) or (i == len(items) - 1)

            if should_split and current_keys:
                leaf = Node(is_leaf=True)
                leaf.keys = current_keys
                leaf.values = current_values

                # Validate leaf before adding
                if self.validate:
                    leaf.validate(self.store, context="_build_leaves")

                leaves.append(leaf)

                # Reset for next leaf
                current_keys = []
                current_values = []
                roll_hash = self.seed  # Reset hash for next node

        return leaves if leaves else [Node(is_leaf=True)]

    def _print_tree(self, label: str = "", verbose: bool = False):
        """
        Print tree structure for debugging.

        Args:
            label: Label to display with the tree
            verbose: If True, print all leaf node values. If False, only show first/last keys and count.
        """
        print(f"\n{'='*60}")
        print(f"TREE {label}:")
        print(f"{'='*60}")
        # For the root, we don't have a hash readily available
        # We'd need to compute it or track it separately
        root_hash = self._hash_node(self.root)
        self._print_node(self.root, root_hash, prefix="", is_last=True, verbose=verbose)

    def _print_node(self, node: Node, node_hash: Optional[str], prefix: str = "", is_last: bool = True, reused_hashes: Optional[set[str]] = None, verbose: bool = False):
        """
        Recursively print node and its children.

        Args:
            node: Node to print
            node_hash: Hash of the node
            prefix: Prefix for tree formatting
            is_last: Whether this is the last child
            reused_hashes: Set of reused hashes to mark
            verbose: If True, print all leaf node values. If False, only show first/last keys and count.
        """
        branch = "└── " if is_last else "├── "

        # Check if this node was reused
        reused_flag = ""
        if reused_hashes is not None and node_hash is not None and node_hash in reused_hashes:
            reused_flag = " <- REUSED!"

        if node.is_leaf:
            hash_str = f"#{node_hash}" if node_hash is not None else "#root"
            if verbose:
                # Show all key-value pairs
                data = list(zip(node.keys, node.values))
                print(f"{prefix}{branch}LEAF {hash_str}: {data}{reused_flag}")
            else:
                # Show only first and last keys, and the count
                count = len(node.keys)
                if count == 0:
                    print(f"{prefix}{branch}LEAF {hash_str}: (empty){reused_flag}")
                elif count == 1:
                    print(f"{prefix}{branch}LEAF {hash_str}: [{node.keys[0]}] (1 key){reused_flag}")
                else:
                    first_key = node.keys[0]
                    last_key = node.keys[-1]
                    print(f"{prefix}{branch}LEAF {hash_str}: [{first_key} ... {last_key}] ({count} keys){reused_flag}")
        else:
            hash_str = f"#{node_hash}" if node_hash is not None else "#root"
            print(f"{prefix}{branch}INTERNAL {hash_str}: keys={node.keys}{reused_flag}")

            # Print children
            extension = "    " if is_last else "│   "
            for i, child_hash in enumerate(node.values):
                child = self._get_node(child_hash)
                if child is None:
                    continue
                child_is_last = (i == len(node.values) - 1)
                self._print_node(child, child_hash, prefix + extension, child_is_last, reused_hashes, verbose)

    def _print_ops(self):
        """Print operation statistics"""
        print(f"\n{'='*60}")
        print("OPERATIONS:")
        print(f"{'='*60}")
        stats = self._summarize_ops()
        for key, value in stats.items():
            print(f"{key}: {value}")

    def _summarize_ops(self) -> dict[str, int]:
        """Summarize operations into statistics"""
        return self.stats.to_dict()

    def items(self, prefix: str = "") -> Iterator[tuple[str, str]]:
        """
        Generator that yields (key, value) pairs with keys matching the given prefix.

        Uses TreeCursor to iterate through all items and filters by prefix.
        When a prefix is provided, seeks to the prefix for O(log n) performance
        (if the tree has valid separator invariants).

        Args:
            prefix: Key prefix to filter (default: "" returns all items)

        Yields:
            Tuples of (key, value) for keys matching the prefix
        """
        # Get root hash
        root_hash = self._hash_node(self.root)

        # Create cursor, seeking to prefix if provided
        cursor = TreeCursor(self.store, root_hash, seek_to=prefix if prefix else None)
        entry = cursor.next()

        # Track whether we've found any matches
        # This is needed for trees with invalid separator invariants where seek may fail
        found_match = False

        while entry:
            key, value = entry
            # Check if key matches prefix
            if isinstance(key, str):
                if key.startswith(prefix):
                    found_match = True
                    yield (key, value)
                elif prefix and found_match:
                    # Key doesn't match prefix and we've seen matches before
                    # Since keys are sorted, we're past all matches
                    break
            else:
                # Non-string keys - only yield if no prefix
                if not prefix:
                    yield (key, value)

            entry = cursor.next()

    def validate_sorted(self) -> tuple[bool, Optional[str], Optional[int], Optional[str], Optional[str]]:
        """
        Validate that all keys in the tree are in sorted order with no duplicates.

        Returns:
            Tuple of (is_valid, error_message, position, prev_key, current_key)
            If valid: (True, None, None, None, None)
            If invalid: (False, error_msg, position, prev_key, current_key)
        """
        root_hash = self._hash_node(self.root)
        cursor = TreeCursor(self.store, root_hash)

        prev_key = None
        position = 0

        entry = cursor.next()
        while entry:
            position += 1
            key = entry[0]

            if prev_key is not None:
                if key == prev_key:
                    return (False, f"Duplicate key at position {position}", position, prev_key, key)
                elif key < prev_key:
                    return (False, f"Keys out of order at position {position}", position, prev_key, key)

            prev_key = key
            entry = cursor.next()

        return (True, None, None, None, None)
