"""
Tests for ProllyTree implementation.
"""

import pytest
from prollypy.tree import ProllyTree
from prollypy.store import MemoryStore


@pytest.fixture
def empty_tree():
    """Create an empty tree with test parameters."""
    return ProllyTree(pattern=0.0001, seed=42)


def _do_insert(old_tree, mutations, expected_contents, verbose=False):
    """
    Helper function to test batch insert (functional style).

    Args:
        old_tree: Current ProllyTree instance (will not be modified)
        mutations: List of (key, value) tuples to insert
        expected_contents: Expected list of (key, value) after insert
        verbose: Whether to print detailed output

    Returns:
        (new_tree, stats): New ProllyTree instance and operation statistics
    """
    # Capture existing node hashes before insert (if using MemoryStore)
    old_node_hashes = set()
    if isinstance(old_tree.store, MemoryStore):
        old_node_hashes = set(old_tree.store.nodes.keys())

    if verbose:
        print(f"\n{'-'*60}")
        print(f"INSERTING: {mutations}")
        print(f"{'-'*60}")
        print("\nTREE BEFORE INSERT:")
        root_hash = old_tree._hash_node(old_tree.root)
        old_tree._print_node(old_tree.root, root_hash, prefix="", is_last=True)

    # Create a new tree that shares the same store
    new_tree = ProllyTree(pattern=old_tree.pattern / (2**32), seed=old_tree.seed, store=old_tree.store)
    new_tree.root = old_tree.root  # Share the root (immutable)

    # Perform the insert (this will create new nodes but won't modify old ones)
    stats = new_tree.insert_batch(mutations, verbose=verbose)

    # Verify the result
    result = list(new_tree.items())
    assert result == expected_contents, f"Expected {expected_contents}, got {result}"

    if verbose:
        print("\nTREE AFTER INSERT:")
        root_hash = new_tree._hash_node(new_tree.root)
        new_tree._print_node(new_tree.root, root_hash, prefix="", is_last=True, reused_hashes=old_node_hashes)

        print(f"\n{'-'*60}")
        print("OPERATION STATS:")
        print(f"{'-'*60}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return new_tree, stats


def test_insert_into_empty_tree(empty_tree):
    """Test inserting batch into empty tree."""
    _, stats = _do_insert(
        empty_tree,
        mutations=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
        expected_contents=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
    )
    assert stats['nodes_created'] > 0


def test_insert_interleaved_keys(empty_tree):
    """Test inserting batch with interleaved keys."""
    tree1, _ = _do_insert(
        empty_tree,
        mutations=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
        expected_contents=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
    )

    _, stats = _do_insert(
        tree1,
        mutations=[(i, f"v{i}") for i in [1, 3, 5, 7, 9, 11]],
        expected_contents=[(i, f"v{i}") for i in range(1, 13)],
    )
    assert stats['nodes_created'] > 0


def test_insert_unaffected_range(empty_tree):
    """Test inserting batch with keys in unaffected range."""
    tree1, _ = _do_insert(
        empty_tree,
        mutations=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
        expected_contents=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
    )

    tree2, _ = _do_insert(
        tree1,
        mutations=[(i, f"v{i}") for i in [1, 3, 5, 7, 9, 11]],
        expected_contents=[(i, f"v{i}") for i in range(1, 13)],
    )

    # Insert keys > 12, which should only affect the right subtree
    _, stats = _do_insert(
        tree2,
        mutations=[(i, f"v{i}") for i in [13, 14, 15, 16]],
        expected_contents=[(i, f"v{i}") for i in range(1, 17)],
    )
    # Note: subtree reuse depends on how splits occur with the rolling hash
    assert stats['nodes_created'] > 0


def test_large_insert_multiple_splits(empty_tree):
    """Test large insert causing multiple splits."""
    tree1, _ = _do_insert(
        empty_tree,
        mutations=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
        expected_contents=[(i, f"v{i}") for i in [2, 4, 6, 8, 10, 12]],
    )

    tree2, _ = _do_insert(
        tree1,
        mutations=[(i, f"v{i}") for i in [1, 3, 5, 7, 9, 11]],
        expected_contents=[(i, f"v{i}") for i in range(1, 13)],
    )

    tree3, _ = _do_insert(
        tree2,
        mutations=[(i, f"v{i}") for i in [13, 14, 15, 16]],
        expected_contents=[(i, f"v{i}") for i in range(1, 17)],
    )

    # Insert many more keys to cause internal nodes to split
    _, stats = _do_insert(
        tree3,
        mutations=[(i, f"v{i}") for i in range(17, 41)],  # Add 24 more keys (17-40)
        expected_contents=[(i, f"v{i}") for i in range(1, 41)],
    )
    assert stats['nodes_created'] > 0


def test_separator_invariants_simple():
    """Test that separator invariants hold for a simple tree."""
    store = MemoryStore()
    tree = ProllyTree(pattern=0.0001, seed=42, store=store, validate=True)

    # Insert enough data to create internal nodes
    data = [(f'key{i:04d}', f'value{i}') for i in range(100)]
    tree.insert_batch(data, verbose=False)

    # Validate should catch any separator violations
    tree.root.validate(store, context="test_separator_invariants_simple")

    # Explicitly check separator invariants
    _verify_separator_invariants(tree.root, store)


def test_separator_invariants_after_mutations():
    """Test that separator invariants hold after mutations."""
    store = MemoryStore()
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store, validate=True)

    # Build initial tree
    data1 = [(f'key{i:04d}', f'value{i}') for i in range(100)]
    tree1.insert_batch(data1, verbose=False)

    # Verify invariants
    _verify_separator_invariants(tree1.root, store)

    # Mutate the tree
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store, validate=True)
    tree2.root = tree1.root

    data2 = [(f'key{i:04d}', f'UPDATED{i}') for i in range(0, 100, 10)]
    tree2.insert_batch(data2, verbose=False)

    # Verify invariants still hold after mutations
    _verify_separator_invariants(tree2.root, store)


def test_separator_invariants_large_tree():
    """Test separator invariants on a larger tree."""
    store = MemoryStore()
    tree = ProllyTree(pattern=0.0001, seed=42, store=store, validate=True)

    # Insert enough data to create deep tree with multiple levels
    data = [(f'key{i:05d}', f'value{i}') for i in range(500)]
    tree.insert_batch(data, verbose=False)

    # Verify invariants
    _verify_separator_invariants(tree.root, store)


def _verify_separator_invariants(node, store):
    """
    Recursively verify separator invariants for entire tree.

    For each internal node, verifies that:
    - keys[i] equals the first key in child values[i+1]
    - All keys in values[i] are in the correct range based on separators
    """
    if node.is_leaf:
        return

    # Check each separator
    for i, separator in enumerate(node.keys):
        # separator should equal first key in child i+1
        child_idx = i + 1
        assert child_idx < len(node.values), f"Separator {i} has no corresponding child"

        child_hash = node.values[child_idx]
        child = store.get_node(child_hash)
        assert child is not None, f"Child {child_idx} not found in store"

        # Get first key from child
        first_key = _get_first_key(child, store)
        assert first_key is not None, f"Child {child_idx} has no keys"

        assert separator == first_key, (
            f"Separator invariant violated:\n"
            f"  Separator index: {i}\n"
            f"  Expected (first key of child {child_idx}): {first_key}\n"
            f"  Actual separator: {separator}"
        )

    # Verify all keys in each child are in the correct range
    for i, child_hash in enumerate(node.values):
        child = store.get_node(child_hash)
        assert child is not None

        # Determine the valid key range for this child
        if i == 0:
            # First child: all keys < keys[0]
            if len(node.keys) > 0:
                _verify_all_keys_less_than(child, store, node.keys[0])
        elif i < len(node.keys):
            # Middle child: keys[i-1] <= key < keys[i]
            _verify_all_keys_in_range(child, store, node.keys[i-1], node.keys[i])
        else:
            # Last child: keys[-1] <= key
            _verify_all_keys_gte(child, store, node.keys[-1])

        # Recursively check child
        _verify_separator_invariants(child, store)


def _get_first_key(node, store):
    """Get the first key in a node's subtree."""
    if node.is_leaf:
        return node.keys[0] if len(node.keys) > 0 else None
    else:
        if len(node.values) == 0:
            return None
        child_hash = node.values[0]
        child = store.get_node(child_hash)
        if child is None:
            return None
        return _get_first_key(child, store)


def _verify_all_keys_less_than(node, store, upper_bound):
    """Verify all keys in subtree are < upper_bound."""
    if node.is_leaf:
        for key in node.keys:
            assert key < upper_bound, f"Key {key} should be < {upper_bound}"
    else:
        for child_hash in node.values:
            child = store.get_node(child_hash)
            _verify_all_keys_less_than(child, store, upper_bound)


def _verify_all_keys_in_range(node, store, lower_bound, upper_bound):
    """Verify all keys in subtree are in [lower_bound, upper_bound)."""
    if node.is_leaf:
        for key in node.keys:
            assert lower_bound <= key < upper_bound, (
                f"Key {key} should be in [{lower_bound}, {upper_bound})"
            )
    else:
        for child_hash in node.values:
            child = store.get_node(child_hash)
            _verify_all_keys_in_range(child, store, lower_bound, upper_bound)


def _verify_all_keys_gte(node, store, lower_bound):
    """Verify all keys in subtree are >= lower_bound."""
    if node.is_leaf:
        for key in node.keys:
            assert key >= lower_bound, f"Key {key} should be >= {lower_bound}"
    else:
        for child_hash in node.values:
            child = store.get_node(child_hash)
            _verify_all_keys_gte(child, store, lower_bound)
