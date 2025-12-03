"""
ProllyTree implementation with content-based splitting.

Features:
- Content-addressed nodes (hash of contents)
- Rolling hash-based splitting (Rabin fingerprinting)
- Incremental batch insert with subtree reuse
- Pluggable storage backends via BlockStore protocol
"""

import hashlib
from typing import Optional, Iterator
from .node import Node
from .store import BlockStore, MemoryBlockStore, CachedFSBlockStore
from .cursor import TreeCursor

# Chunking constants
MIN_CHUNK_SIZE = 2      # Minimum items per node (never split below this)
MAX_CHUNK_SIZE = 1024   # Maximum items per node (always split at this)
TARGET_CHUNK_SIZE = 256 # Threshold doubles every this many items (higher = less aggressive ramping)


class ProllyTree:
    def __init__(self, pattern=0.01, seed=42, store: Optional[BlockStore] = None, validate=False):
        """
        Initialize ProllyTree with content-based splitting.

        Args:
            pattern: Split probability (0.0 to 1.0). Higher = smaller nodes, wider trees.
                    Default 0.01 means ~100 entries per node on average.
            seed: Seed for rolling hash function for reproducibility
            store: Storage backend (defaults to MemoryBlockStore if not provided)
            validate: If True, validate node structure during tree building (slower)
        """
        self.pattern = int(pattern * (2**32))  # Convert to uint32 threshold
        self.seed = seed
        self.store = store if store is not None else MemoryBlockStore()
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

    def _should_split(self, key: bytes, current_size: int) -> bool:
        """
        Determine if we should split after this item using size-aware threshold.

        The threshold grows with chunk size: starts at self.pattern, doubles every
        TARGET_CHUNK_SIZE items. This ensures:
        - Small chunks: low split probability (controlled by pattern)
        - Large chunks: increasing split probability
        - MAX_CHUNK_SIZE: forced split

        Maintains insertion-order independence since decision only depends on:
        - The key's hash (deterministic)
        - Current chunk size (same for same key sequence)
        """
        if current_size < MIN_CHUNK_SIZE:
            return False
        if current_size >= MAX_CHUNK_SIZE:
            return True

        # Convert string to bytes if needed
        key_bytes = key.encode('utf-8') if isinstance(key, str) else key
        key_hash = self._rolling_hash(self.seed, key_bytes)

        # Threshold grows exponentially with size
        # At size=TARGET_CHUNK_SIZE, threshold = pattern * 2
        # At size=2*TARGET_CHUNK_SIZE, threshold = pattern * 4
        size_factor = 2 ** (current_size / TARGET_CHUNK_SIZE)
        adjusted_threshold = int(min(self.pattern * size_factor, 2**31))  # Cap at 50%

        return key_hash < adjusted_threshold

    def _hash_node(self, node: Node) -> bytes:
        """
        Compute content hash for a node.

        For leaf nodes: hash the key-value pairs
        For internal nodes: hash the (separator_key, child_hash) pairs
        """
        content = []
        if node.is_leaf:
            # Hash key-value pairs
            for key, value in zip(node.keys, node.values):
                content.append(key + b':' + value)
        else:
            # Hash separator keys and child hashes
            for i, child_hash in enumerate(node.values):
                if i < len(node.keys):
                    content.append(node.keys[i] + b':' + child_hash)
                else:
                    content.append(b'_:' + child_hash)

        # Combine all content and hash
        combined = b'|'.join(content)
        return hashlib.sha256(combined).digest()[:16]  # Use first 16 bytes

    def _store_node(self, node: Node) -> bytes:
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

    def _get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve node by hash"""
        return self.store.get_node(node_hash)

    def insert_batch(self, mutations: list[tuple[bytes, bytes]], verbose: bool = True) -> dict[str, int]:
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

        # Track cache stats before this batch (if using CachedFSBlockStore)
        cache_stats_before = None
        if isinstance(self.store, CachedFSBlockStore):
            cache_stats_before = {
                'hits': self.store.cache_hits,
                'misses': self.store.cache_misses
            }

        # Rebuild tree with mutations (always quiet during rebuild)
        result_nodes = self._rebuild_with_mutations(self.root, mutations, verbose=False)

        # If we got multiple nodes back, build an internal node structure
        if len(result_nodes) == 0:
            new_root = Node(is_leaf=True)
        elif len(result_nodes) == 1:
            new_root = result_nodes[0]
        else:
            new_root = self._build_internal_from_children(result_nodes, verbose=False)

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

            # Add cache stats if using CachedFSBlockStore
            if isinstance(self.store, CachedFSBlockStore) and cache_stats_before:
                hits_delta = self.store.cache_hits - cache_stats_before['hits']
                misses_delta = self.store.cache_misses - cache_stats_before['misses']
                summary_parts.append(f"{hits_delta} cache hits")
                summary_parts.append(f"{misses_delta} cache misses")

                # Add average node sizes from CachedFSBlockStore
                size_stats = self.store.get_size_stats()
                if size_stats['avg_leaf_size'] > 0:
                    summary_parts.append(f"avg leaf: {size_stats['avg_leaf_size']:.0f}B")
                if size_stats['avg_internal_size'] > 0:
                    summary_parts.append(f"avg internal: {size_stats['avg_internal_size']:.0f}B")

            print("; ".join(summary_parts))

        return stats

    def _collect_all_items(self, node: Node) -> list[tuple[bytes, bytes]]:
        """
        Collect all key-value pairs from a node's subtree.
        Used when we need to rebuild from scratch.
        """
        if node.is_leaf:
            return list(zip(node.keys, node.values))
        else:
            items = []
            for child_hash in node.values:
                child = self._get_node(child_hash)
                if child is not None:
                    items.extend(self._collect_all_items(child))
            return items

    def _rebuild_with_mutations(self, node: Node, mutations: list[tuple[bytes, bytes]], verbose: bool = True) -> list[Node]:
        """
        Core incremental rebuild logic.

        For insertion-order independence, when mutations affect a subtree,
        we collect all items from that subtree and rebuild from scratch.
        This ensures the same structure regardless of insertion order.

        Returns: list of nodes (possibly multiple if the node split)
        """
        if not mutations:
            # No mutations for this subtree - REUSE it!
            return [node]

        # Collect all existing items from this subtree
        existing_items = self._collect_all_items(node)

        # Merge with mutations
        merged = self._merge_sorted(existing_items, mutations)

        # Rebuild leaves from the merged items
        return self._build_leaves(merged)

    def _get_first_key(self, node: Node) -> Optional[bytes]:
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
        Build an internal tree from a list of children using per-child boundary detection.

        Each child independently decides if it's a boundary based on its first key's hash.
        A child is a boundary (last child in a node) if _should_split returns True for
        the child's first key. This ensures insertion-order independence since split
        decisions only depend on the key and current chunk size.

        Args:
            children: List of Node objects

        Returns:
            Node (single child, or newly created internal node/tree)
        """
        if len(children) == 0:
            raise ValueError("Cannot build internal node with no children")

        if len(children) == 1:
            # Single child - just return it (no need for parent)
            return children[0]

        # Build internal nodes using size-aware boundary detection
        internal_nodes = []
        current_children = []

        for i, child in enumerate(children):
            current_children.append(child)

            # Get boundary key for this child (first key in subtree)
            boundary_key = self._get_first_key(child)

            # Use size-aware splitting, or force split on last child
            is_last = (i == len(children) - 1)
            if boundary_key is not None:
                should_split = self._should_split(boundary_key, len(current_children)) or is_last
            else:
                should_split = is_last  # No key, only split if last

            if should_split and current_children:
                # Create internal node from current children
                internal_node = Node(is_leaf=False)

                for j, c in enumerate(current_children):
                    # Store or reuse child hash
                    if hasattr(c, '_reused_hash'):
                        node_hash = getattr(c, '_reused_hash')
                        delattr(c, '_reused_hash')
                    else:
                        node_hash = self._store_node(c)

                    internal_node.values.append(node_hash)

                    # Add separator key (first key of next child)
                    if j < len(current_children) - 1:
                        next_child = current_children[j + 1]
                        separator = self._get_first_key(next_child)
                        if separator is not None:
                            internal_node.keys.append(separator)

                # Validate before adding
                if self.validate:
                    internal_node.validate(self.store, context=f"_build_internal_from_children")

                internal_nodes.append(internal_node)

                # Reset for next internal node
                current_children = []

        # If we only created one internal node with a single child, unwrap it
        if len(internal_nodes) == 1:
            node = internal_nodes[0]
            if len(node.values) == 1:
                node_hash = node.values[0]
                child = self._get_node(node_hash)
                if child is None:
                    raise ValueError(f"Child node {node_hash} not found in store")
                return child
            return node

        # Multiple internal nodes - recursively build parent level
        return self._build_internal_from_children(internal_nodes, verbose)

    def _merge_sorted(self, old_items: list[tuple[bytes, bytes]], new_items: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
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

    def _build_leaves(self, items: list[tuple[bytes, bytes]]) -> list[Node]:
        """
        Build leaf nodes from sorted items using size-aware boundary detection.

        Each item independently decides if it's a boundary based on its own hash
        and the current chunk size. Split probability increases as chunks grow.
        This ensures insertion-order independence since split decisions only
        depend on the key and current chunk size.
        """
        if not items:
            return []

        leaves = []
        current_keys = []
        current_values = []

        for i, (key, value) in enumerate(items):
            current_keys.append(key)
            current_values.append(value)

            # Use size-aware splitting, or force split on last item
            is_last = (i == len(items) - 1)
            should_split = self._should_split(key, len(current_keys)) or is_last

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

    def _print_node(self, node: Node, node_hash: Optional[bytes], prefix: str = "", is_last: bool = True, reused_hashes: Optional[set[bytes]] = None, verbose: bool = False):
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
            hash_str = f"#{node_hash.hex()}" if node_hash is not None else "#root"
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
            hash_str = f"#{node_hash.hex()}" if node_hash is not None else "#root"
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

    def items(self, prefix: bytes = b"") -> Iterator[tuple[bytes, bytes]]:
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
            if key.startswith(prefix):
                found_match = True
                yield (key, value)
            elif prefix and found_match:
                # Key doesn't match prefix and we've seen matches before
                # Since keys are sorted, we're past all matches
                break

            entry = cursor.next()

    def validate_sorted(self) -> tuple[bool, Optional[str], Optional[int], Optional[bytes], Optional[bytes]]:
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
