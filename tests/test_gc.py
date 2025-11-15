"""
Tests for garbage collection.
"""

import pytest
from prollypy.tree import ProllyTree
from prollypy.store import MemoryStore
from prollypy.store_gc import (
    find_reachable_nodes,
    find_garbage_nodes,
    collect_garbage_stats,
    garbage_collect,
    remove_garbage
)


@pytest.fixture
def store():
    """Create a memory store for testing."""
    return MemoryStore()


def test_single_tree_no_garbage(store):
    """Test that a single tree has no garbage."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(100)], verbose=False)

    root_hash = tree._hash_node(tree.root)

    # Find garbage
    garbage = find_garbage_nodes(store, {root_hash})

    assert len(garbage) == 0, "Single tree should have no garbage"


def test_two_trees_sharing_nodes(store):
    """Test two trees that share some nodes."""
    # Create first tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(100)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create second tree by modifying first tree (should share most nodes)
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree1.root
    tree2.insert_batch([(b'0', b'modified')], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Both trees should have no garbage
    garbage_both = find_garbage_nodes(store, {hash1, hash2})
    assert len(garbage_both) == 0

    # If we only keep tree2, tree1's unique nodes become garbage
    garbage_tree2_only = find_garbage_nodes(store, {hash2})
    assert len(garbage_tree2_only) > 0, "Should have garbage from tree1"


def test_old_tree_becomes_garbage(store):
    """Test that old tree versions become garbage when not kept."""
    # Create initial tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)
    nodes_after_tree1 = store.count_nodes()

    # Create modified tree
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree1.root
    tree2.insert_batch([(str(i).encode(), f'modified{i}'.encode()) for i in range(10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)
    nodes_after_tree2 = store.count_nodes()

    # New nodes were created
    assert nodes_after_tree2 > nodes_after_tree1

    # If we keep both roots, no garbage
    garbage_both = find_garbage_nodes(store, {hash1, hash2})
    assert len(garbage_both) == 0

    # If we only keep tree2, tree1's unique nodes are garbage
    garbage_tree2_only = find_garbage_nodes(store, {hash2})
    assert len(garbage_tree2_only) > 0


def test_reachable_nodes_traversal(store):
    """Test that find_reachable_nodes correctly traverses tree."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(100)], verbose=False)

    root_hash = tree._hash_node(tree.root)

    # Find reachable nodes
    reachable = find_reachable_nodes(store, {root_hash})

    # Should equal total nodes in store (single tree, no garbage)
    assert len(reachable) == store.count_nodes()

    # Root should be in reachable set
    assert root_hash in reachable


def test_gc_stats_empty_store(store):
    """Test GC stats on empty store."""
    stats = collect_garbage_stats(store, set())

    assert stats.total_nodes == 0
    assert stats.reachable_nodes == 0
    assert stats.garbage_nodes == 0
    assert stats.reachable_percent == 0.0
    assert stats.garbage_percent == 0.0


def test_gc_stats_single_tree(store):
    """Test GC stats on single tree."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(100)], verbose=False)
    root_hash = tree._hash_node(tree.root)

    stats = collect_garbage_stats(store, {root_hash})

    assert stats.total_nodes == store.count_nodes()
    assert stats.reachable_nodes == stats.total_nodes
    assert stats.garbage_nodes == 0
    assert stats.reachable_percent == 100.0
    assert stats.garbage_percent == 0.0


def test_gc_stats_with_garbage(store):
    """Test GC stats when there is garbage."""
    # Create tree1
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 (modified version)
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree1.root
    tree2.insert_batch([(str(i).encode(), f'modified{i}'.encode()) for i in range(10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Get stats keeping only tree2
    stats = collect_garbage_stats(store, {hash2})

    assert stats.total_nodes == store.count_nodes()
    assert stats.reachable_nodes < stats.total_nodes
    assert stats.garbage_nodes > 0
    assert stats.garbage_nodes == stats.total_nodes - stats.reachable_nodes
    assert stats.reachable_percent < 100.0
    assert stats.garbage_percent > 0.0


def test_garbage_collect_dry_run(store):
    """Test garbage collection dry run doesn't remove anything."""
    # Create and modify tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree1.root
    tree2.insert_batch([(str(i).encode(), f'modified{i}'.encode()) for i in range(10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    initial_count = store.count_nodes()

    # Dry run GC
    stats = garbage_collect(store, {hash2}, dry_run=True)

    # Nothing should be removed
    assert store.count_nodes() == initial_count
    assert stats.garbage_nodes > 0  # But we know there is garbage


def test_garbage_collect_live(store):
    """Test actual garbage collection removes nodes."""
    # Create and modify tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree1.root
    tree2.insert_batch([(str(i).encode(), f'modified{i}'.encode()) for i in range(10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    initial_count = store.count_nodes()

    # Live GC (keep only tree2)
    stats = garbage_collect(store, {hash2}, dry_run=False)

    # Nodes should be removed
    final_count = store.count_nodes()
    assert final_count < initial_count
    assert final_count == stats.reachable_nodes
    assert initial_count - final_count == stats.garbage_nodes


def test_multiple_roots_preserved(store):
    """Test that multiple roots are all preserved."""
    # Create three independent trees
    trees = []
    hashes = []

    for i in range(3):
        tree = ProllyTree(pattern=0.0001, seed=42, store=store)
        tree.insert_batch([(str(j + i*100).encode(), f'tree{i}_v{j}'.encode()) for j in range(20)], verbose=False)
        hash_val = tree._hash_node(tree.root)
        trees.append(tree)
        hashes.append(hash_val)

    # Keep all three roots
    stats = collect_garbage_stats(store, set(hashes))

    # No garbage when all roots are kept
    assert stats.garbage_nodes == 0
    assert stats.reachable_nodes == stats.total_nodes


def test_empty_root_set(store):
    """Test GC with empty root set (everything is garbage)."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)

    total_nodes = store.count_nodes()

    # Empty root set means everything is garbage
    stats = collect_garbage_stats(store, set())

    assert stats.total_nodes == total_nodes
    assert stats.reachable_nodes == 0
    assert stats.garbage_nodes == total_nodes
    assert stats.garbage_percent == 100.0


def test_nonexistent_root_hash(store):
    """Test GC with nonexistent root hash."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)

    total_nodes = store.count_nodes()

    # Use a nonexistent hash
    fake_hash = "0000000000000000"
    stats = collect_garbage_stats(store, {fake_hash})

    # All nodes should be garbage (fake root doesn't exist)
    assert stats.garbage_nodes == total_nodes
    assert stats.reachable_nodes == 0


def test_remove_garbage_function(store):
    """Test the remove_garbage function directly."""
    # Create tree
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)
    hash1 = tree._hash_node(tree.root)

    # Modify tree
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree.root
    tree2.insert_batch([(str(i).encode(), f'modified{i}'.encode()) for i in range(10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Find garbage
    garbage = find_garbage_nodes(store, {hash2})
    initial_count = store.count_nodes()

    # Remove garbage
    removed_count = remove_garbage(store, garbage)

    assert removed_count == len(garbage)
    assert store.count_nodes() == initial_count - removed_count


def test_gc_preserves_reachable_data(store):
    """Test that GC preserves all reachable data."""
    # Create tree
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    data = [(str(i).encode(), f'value{i}'.encode()) for i in range(100)]
    # Sort data since byte strings sort lexicographically (b'9' > b'10')
    data.sort(key=lambda x: x[0])
    tree.insert_batch(data, verbose=False)
    hash1 = tree._hash_node(tree.root)

    # Modify tree
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree.root
    tree2.insert_batch([(b'50', b'modified')], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # GC keeping only tree2
    garbage_collect(store, {hash2}, dry_run=False)

    # Verify tree2 is still intact and accessible
    tree2_verify = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2_verify.root = store.get_node(hash2)

    # Check that we can still access all keys from tree2
    items = list(tree2_verify.items())
    assert len(items) == 100

    # Verify the modified value is there
    items_dict = dict(items)
    assert items_dict[b'50'] == b'modified'


def test_gc_stats_repr(store):
    """Test GCStats repr formatting."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f'v{i}'.encode()) for i in range(50)], verbose=False)
    hash1 = tree._hash_node(tree.root)

    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.root = tree.root
    tree2.insert_batch([(str(i).encode(), f'modified{i}'.encode()) for i in range(10)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    stats = collect_garbage_stats(store, {hash2})

    # Check repr includes key information
    repr_str = repr(stats)
    assert 'GCStats' in repr_str
    assert 'total=' in repr_str
    assert 'reachable=' in repr_str
    assert 'garbage=' in repr_str
    assert '%' in repr_str
