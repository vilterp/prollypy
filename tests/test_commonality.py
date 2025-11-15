"""
Tests for commonality analysis.
"""

import pytest
from prollypy.tree import ProllyTree
from prollypy.store import MemoryBlockStore
from prollypy.commonality import compute_commonality, collect_node_hashes


@pytest.fixture
def shared_store():
    """Create a shared memory store for tests."""
    return MemoryBlockStore()


def test_identical_trees(shared_store):
    """Test commonality of two identical trees."""
    # Create tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(1, 11)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Same tree, same hash
    stats = compute_commonality(shared_store, hash1, hash1)

    assert stats['left_total'] == stats['right_total']
    assert stats['both_count'] == stats['left_total']
    assert len(stats['left_only']) == 0
    assert len(stats['right_only']) == 0


def test_completely_disjoint_trees(shared_store):
    """Test commonality of two completely different trees."""
    # Create first tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(1, 11)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create second tree with different data
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree2.insert_batch([(str(i+100).encode(), f'v{i+100}'.encode()) for i in range(1, 11)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    stats = compute_commonality(shared_store, hash1, hash2)

    # Should have no shared nodes (different data means different hashes)
    assert stats['both_count'] == 0
    assert len(stats['left_only']) == stats['left_total']
    assert len(stats['right_only']) == stats['right_total']


def test_overlapping_trees(shared_store):
    """Test commonality statistics are computed correctly."""
    # Create first tree with keys 1-10
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(1, 11)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create second tree with different keys
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree2.insert_batch([(str(i).encode(), f'w{i}'.encode()) for i in range(1, 11)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    stats = compute_commonality(shared_store, hash1, hash2)

    # Verify the counts are consistent
    assert stats['left_total'] == len(stats['left_only']) + stats['both_count']
    assert stats['right_total'] == len(stats['right_only']) + stats['both_count']
    # At least one tree should have nodes
    assert stats['left_total'] > 0
    assert stats['right_total'] > 0


def test_subset_tree(shared_store):
    """Test commonality with different sized trees."""
    # Create first tree with keys 1-5
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(1, 6)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create second tree with more keys
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree2.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(1, 21)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    stats = compute_commonality(shared_store, hash1, hash2)

    # Verify counts are valid
    assert stats['left_total'] > 0
    assert stats['right_total'] > 0
    assert stats['left_total'] == len(stats['left_only']) + stats['both_count']
    assert stats['right_total'] == len(stats['right_only']) + stats['both_count']


def test_collect_node_hashes(shared_store):
    """Test that collect_node_hashes visits all nodes."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=shared_store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(1, 51)], verbose=False)
    root_hash = tree._hash_node(tree.root)

    # Collect all node hashes
    hashes = collect_node_hashes(shared_store, root_hash)

    # Should have at least the root
    assert len(hashes) >= 1
    assert root_hash in hashes

    # All collected hashes should exist in store
    for h in hashes:
        node = shared_store.get_node(h)
        assert node is not None
