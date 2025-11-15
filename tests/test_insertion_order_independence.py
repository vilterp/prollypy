"""
Tests for insertion order independence.

ProllyTree should produce identical tree structures regardless of the order
in which keys are inserted, updated, or deleted. This is a critical property
for ensuring deterministic diffs.
"""

import pytest
import random

from prollypy.tree import ProllyTree
from prollypy.store import MemoryStore


@pytest.fixture
def store():
    """Create a shared MemoryStore for testing."""
    return MemoryStore()


def test_insertion_order_independence_simple(store):
    """
    Test that inserting batches in different orders produces identical trees.

    Each batch must be sorted, but the batches themselves can be inserted
    in any order and should produce the same final tree structure.
    """
    # Split data into 3 batches
    batch1 = [(1, "a"), (2, "b")]
    batch2 = [(3, "c"), (4, "d")]
    batch3 = [(5, "e"), (6, "f")]

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
    keys = [(i, f"value_{i}") for i in range(100)]

    # Split into 4 batches
    batch1 = [(i, f"value_{i}") for i in range(0, 25)]
    batch2 = [(i, f"value_{i}") for i in range(25, 50)]
    batch3 = [(i, f"value_{i}") for i in range(50, 75)]
    batch4 = [(i, f"value_{i}") for i in range(75, 100)]

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
    all_keys = [(i, f"v{i}") for i in range(50)]

    # Insert all at once
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted(all_keys), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Insert in two batches
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    batch1 = [(i, f"v{i}") for i in range(25)]
    batch2 = [(i, f"v{i}") for i in range(25, 50)]
    tree2.insert_batch(sorted(batch1), verbose=False)
    tree2.insert_batch(sorted(batch2), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Insert in three batches with different split points
    tree3 = ProllyTree(pattern=0.0001, seed=42, store=store)
    batch_a = [(i, f"v{i}") for i in range(10)]
    batch_b = [(i, f"v{i}") for i in range(10, 35)]
    batch_c = [(i, f"v{i}") for i in range(35, 50)]
    tree3.insert_batch(sorted(batch_a), verbose=False)
    tree3.insert_batch(sorted(batch_b), verbose=False)
    tree3.insert_batch(sorted(batch_c), verbose=False)
    hash3 = tree3._hash_node(tree3.root)

    assert hash1 == hash2 == hash3, "Trees should have identical hashes regardless of batching strategy"


def test_insertion_order_independence_with_updates(store):
    """Test that updates produce the same result regardless of order."""
    # Initial set of keys
    initial = [(i, f"v{i}") for i in range(20)]

    # Updates to apply
    updates = [(5, "UPDATED_5"), (10, "UPDATED_10"), (15, "UPDATED_15")]

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
        ("/d/table1/row1", "data1"),
        ("/d/table1/row2", "data2"),
        ("/d/table2/row1", "data3"),
        ("/s/schema1", "schema_data"),
        ("/s/schema2", "schema_data2"),
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
    keys = [(i, f"val{i}") for i in range(30)]

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
    store = MemoryStore()
    # Use more keys and higher pattern to ensure splits happen
    keys = [(i, f"value_{i}") for i in range(500)]

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
    keys = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]

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
