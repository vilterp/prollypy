"""
Tests for insertion order independence.

ProllyTree should produce identical tree structures regardless of the order
in which keys are inserted, updated, or deleted. This is a critical property
for ensuring deterministic diffs.
"""

import pytest
import random

from prollypy.tree import ProllyTree
from prollypy.store import MemoryBlockStore


@pytest.fixture
def store():
    """Create a shared MemoryBlockStore for testing."""
    return MemoryBlockStore()


def test_insertion_order_independence_simple(store):
    """
    Test that inserting batches in different orders produces identical trees.

    Each batch must be sorted, but the batches themselves can be inserted
    in any order and should produce the same final tree structure.
    """
    # Split data into 3 batches
    batch1 = [(b"1", b"a"), (b"2", b"b")]
    batch2 = [(b"3", b"c"), (b"4", b"d")]
    batch3 = [(b"5", b"e"), (b"6", b"f")]

    # Insert batches in order: 1, 2, 3
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted(batch1), verbose=False)
    tree1.insert_batch(sorted(batch2), verbose=False)
    tree1.insert_batch(sorted(batch3), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Insert batches in different order: 3, 1, 2
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch(sorted(batch3), verbose=False)
    tree2.insert_batch(sorted(batch1), verbose=False)
    tree2.insert_batch(sorted(batch2), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Insert batches in yet another order: 2, 3, 1
    tree3 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree3.insert_batch(sorted(batch2), verbose=False)
    tree3.insert_batch(sorted(batch3), verbose=False)
    tree3.insert_batch(sorted(batch1), verbose=False)
    hash3 = tree3._hash_node(tree3.root)

    assert hash1 == hash2 == hash3, "Trees should have identical hashes regardless of batch insertion order"


def test_insertion_order_independence_larger(store):
    """
    Test insertion order independence with 100 keys split into multiple batches.

    Insert the same data via different batch orderings.
    """
    # Create 100 key-value pairs
    keys = [(str(i).encode(), f"value_{i}".encode()) for i in range(100)]

    # Split into 4 batches
    batch1 = [(str(i).encode(), f"value_{i}".encode()) for i in range(0, 25)]
    batch2 = [(str(i).encode(), f"value_{i}".encode()) for i in range(25, 50)]
    batch3 = [(str(i).encode(), f"value_{i}".encode()) for i in range(50, 75)]
    batch4 = [(str(i).encode(), f"value_{i}".encode()) for i in range(75, 100)]

    # Insert batches in order: 1, 2, 3, 4
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(batch1, verbose=False)
    tree1.insert_batch(batch2, verbose=False)
    tree1.insert_batch(batch3, verbose=False)
    tree1.insert_batch(batch4, verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Insert batches in reverse order: 4, 3, 2, 1
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch(batch4, verbose=False)
    tree2.insert_batch(batch3, verbose=False)
    tree2.insert_batch(batch2, verbose=False)
    tree2.insert_batch(batch1, verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Insert batches in random order: 3, 1, 4, 2
    tree3 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree3.insert_batch(batch3, verbose=False)
    tree3.insert_batch(batch1, verbose=False)
    tree3.insert_batch(batch4, verbose=False)
    tree3.insert_batch(batch2, verbose=False)
    hash3 = tree3._hash_node(tree3.root)

    assert hash1 == hash2 == hash3, "Trees should have identical hashes regardless of batch insertion order"


def test_insertion_order_independence_multiple_batches(store):
    """Test that inserting in multiple batches produces the same result as single batch."""
    all_keys = [(str(i).encode(), f"v{i}".encode()) for i in range(50)]

    # Insert all at once
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted(all_keys), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Insert in two batches
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    batch1 = [(str(i).encode(), f"v{i}".encode()) for i in range(25)]
    batch2 = [(str(i).encode(), f"v{i}".encode()) for i in range(25, 50)]
    tree2.insert_batch(sorted(batch1), verbose=False)
    tree2.insert_batch(sorted(batch2), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Insert in three batches with different split points
    tree3 = ProllyTree(pattern=0.0001, seed=42, store=store)
    batch_a = [(str(i).encode(), f"v{i}".encode()) for i in range(10)]
    batch_b = [(str(i).encode(), f"v{i}".encode()) for i in range(10, 35)]
    batch_c = [(str(i).encode(), f"v{i}".encode()) for i in range(35, 50)]
    tree3.insert_batch(sorted(batch_a), verbose=False)
    tree3.insert_batch(sorted(batch_b), verbose=False)
    tree3.insert_batch(sorted(batch_c), verbose=False)
    hash3 = tree3._hash_node(tree3.root)

    assert hash1 == hash2 == hash3, "Trees should have identical hashes regardless of batching strategy"


def test_insertion_order_independence_with_updates(store):
    """Test that updates produce the same result regardless of order."""
    # Initial set of keys
    initial = [(str(i).encode(), f"v{i}".encode()) for i in range(20)]

    # Updates to apply
    updates = [(b"5", b"UPDATED_5"), (b"10", b"UPDATED_10"), (b"15", b"UPDATED_15")]

    # Approach 1: Insert all initial, then apply updates
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted(initial), verbose=False)
    tree1.insert_batch(sorted(updates), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Approach 2: Insert with updates already applied
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    final_data = {k: v for k, v in initial}
    for k, v in updates:
        final_data[k] = v
    tree2.insert_batch(sorted(final_data.items()), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    assert hash1 == hash2, "Trees should have identical hashes whether updates are applied separately or merged"


def test_insertion_order_independence_string_keys(store):
    """Test insertion order independence with string keys."""
    keys = [
        (b"/d/table1/row1", b"data1"),
        (b"/d/table1/row2", b"data2"),
        (b"/d/table2/row1", b"data3"),
        (b"/s/schema1", b"schema_data"),
        (b"/s/schema2", b"schema_data2"),
    ]

    # Insert in sorted order
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted(keys), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Insert in different order
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    different_order = [keys[2], keys[4], keys[0], keys[3], keys[1]]
    tree2.insert_batch(sorted(different_order), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    assert hash1 == hash2, "Trees should have identical hashes with string keys"


def test_tree_structure_identical_not_just_hash(store):
    """
    Verify that identical hashes mean identical tree structures, not just hash collisions.

    This test checks the actual tree structure (node keys and values) to ensure
    we're getting true structural identity, not accidental hash matches.
    """
    keys = [(str(i).encode(), f"val{i}".encode()) for i in range(30)]

    # Build tree 1
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted(keys), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Build tree 2 with different insertion order
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    random.seed(456)
    shuffled = keys.copy()
    random.shuffle(shuffled)
    tree2.insert_batch(sorted(shuffled), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Hashes should match
    assert hash1 == hash2

    # Now verify structure is truly identical by comparing nodes
    def compare_nodes(node1, node2):
        """Recursively compare two nodes for structural identity."""
        # Both should be leaves or both should be internal
        assert node1.is_leaf == node2.is_leaf

        # Keys should match
        assert node1.keys == node2.keys

        if node1.is_leaf:
            # Values should match
            assert node1.values == node2.values
        else:
            # For internal nodes, values are child hashes
            assert node1.values == node2.values

            # Recursively compare children
            for child_hash1, child_hash2 in zip(node1.values, node2.values):
                child1 = store.get_node(child_hash1)
                child2 = store.get_node(child_hash2)
                compare_nodes(child1, child2)

    compare_nodes(tree1.root, tree2.root)


def test_insertion_order_independence_different_seeds_fail():
    """
    Verify that trees with DIFFERENT seeds produce DIFFERENT structures.

    This is a negative test to ensure our other tests are actually meaningful -
    if different seeds produced the same structure, our tests wouldn't prove anything.
    """
    store = MemoryBlockStore()
    # Use more keys and higher pattern to ensure splits happen
    keys = [(str(i).encode(), f"value_{i}".encode()) for i in range(500)]

    # Build with seed 42 - use higher pattern to trigger splits
    tree1 = ProllyTree(pattern=0.25, seed=42, store=store)
    tree1.insert_batch(sorted(keys), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Build with different seed
    tree2 = ProllyTree(pattern=0.25, seed=99, store=store)
    tree2.insert_batch(sorted(keys), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # These should be DIFFERENT because seed affects splitting
    assert hash1 != hash2, "Trees with different seeds should have different structures"


def test_incremental_insertion_order_independence(store):
    """Test that inserting keys one by one in different orders produces same result."""
    keys = [(b"1", b"a"), (b"2", b"b"), (b"3", b"c"), (b"4", b"d"), (b"5", b"e")]

    # Insert in ascending order
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    for k, v in keys:
        tree1.insert_batch([(k, v)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Insert in descending order
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    for k, v in reversed(keys):
        tree2.insert_batch([(k, v)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Insert in random order
    tree3 = ProllyTree(pattern=0.0001, seed=42, store=store)
    random_order = [keys[2], keys[0], keys[4], keys[1], keys[3]]
    for k, v in random_order:
        tree3.insert_batch([(k, v)], verbose=False)
    hash3 = tree3._hash_node(tree3.root)

    assert hash1 == hash2 == hash3, "Incremental insertion should be order-independent"


def test_insertion_order_independence_with_splits(store):
    """
    Test insertion order independence when splits actually occur.

    This test uses a higher split probability and more data to ensure
    multiple leaf and internal node splits occur. The tree structure
    should be identical regardless of batch sizes.
    """
    # Use 50% split probability so splits happen frequently
    pattern = 0.5
    num_items = 100

    # Create all items
    all_data = [(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(num_items)]

    # Test 1: Insert all at once
    tree1 = ProllyTree(pattern=pattern, seed=42, store=store)
    tree1.insert_batch(all_data, verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Test 2: Insert in 2 batches
    tree2 = ProllyTree(pattern=pattern, seed=42, store=store)
    half = num_items // 2
    tree2.insert_batch([(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(half)], verbose=False)
    tree2.insert_batch([(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(half, num_items)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Test 3: Insert in 10 batches
    tree3 = ProllyTree(pattern=pattern, seed=42, store=store)
    batch_size = num_items // 10
    for batch_num in range(10):
        start = batch_num * batch_size
        end = start + batch_size
        tree3.insert_batch([(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(start, end)], verbose=False)
    hash3 = tree3._hash_node(tree3.root)

    # Test 4: Insert one-by-one
    tree4 = ProllyTree(pattern=pattern, seed=42, store=store)
    for i in range(num_items):
        tree4.insert_batch([(f'{i:05d}'.encode(), f'value_{i}'.encode())], verbose=False)
    hash4 = tree4._hash_node(tree4.root)

    assert hash1 == hash2, f"Single batch vs 2 batches should produce same tree. Got {hash1.hex()} vs {hash2.hex()}"
    assert hash1 == hash3, f"Single batch vs 10 batches should produce same tree. Got {hash1.hex()} vs {hash3.hex()}"
    assert hash1 == hash4, f"Single batch vs one-by-one should produce same tree. Got {hash1.hex()} vs {hash4.hex()}"


def test_insertion_order_independence_minimal(store):
    """
    Minimal test case for insertion order independence with splits.

    Even with just 10 items and 50% split probability, the tree structure
    should be identical when inserting all at once vs in batches.
    """
    pattern = 0.5

    all_data = [(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(10)]

    # All at once
    tree1 = ProllyTree(pattern=pattern, seed=42, store=store)
    tree1.insert_batch(all_data, verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Two batches of 5
    tree2 = ProllyTree(pattern=pattern, seed=42, store=store)
    tree2.insert_batch([(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(5)], verbose=False)
    tree2.insert_batch([(f'{i:05d}'.encode(), f'value_{i}'.encode()) for i in range(5, 10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    assert hash1 == hash2, f"Trees should be identical. Single batch: {hash1.hex()}, Two batches: {hash2.hex()}"
