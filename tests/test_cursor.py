"""
Tests for TreeCursor.
"""

import pytest
from prollypy.tree import ProllyTree
from prollypy.store import MemoryBlockStore
from prollypy.cursor import TreeCursor


@pytest.fixture
def sample_tree():
    """Create a sample tree with known data."""
    store = MemoryBlockStore()
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)

    # Insert data that will create multiple nodes
    data = [(f'key{i:04d}'.encode(), f'value{i}'.encode()) for i in range(100)]
    tree.insert_batch(data, verbose=False)

    root_hash = tree._hash_node(tree.root)
    return store, root_hash, data


def test_cursor_iterates_all_keys(sample_tree):
    """Test that cursor visits all keys in order."""
    store, root_hash, expected_data = sample_tree

    cursor = TreeCursor(store, root_hash)
    results = []

    entry = cursor.next()
    while entry:
        results.append(entry)
        entry = cursor.next()

    assert len(results) == len(expected_data)
    assert results == expected_data


def test_cursor_seek_to_middle(sample_tree):
    """Test seeking to a key in the middle."""
    store, root_hash, expected_data = sample_tree

    # Seek to key0050
    cursor = TreeCursor(store, root_hash, seek_to=b'key0050')
    results = []

    entry = cursor.next()
    while entry:
        results.append(entry)
        entry = cursor.next()

    # Should get key0050 through key0099 (50 keys)
    assert len(results) == 50
    assert results[0][0] == b'key0050'
    assert results[-1][0] == b'key0099'


def test_cursor_seek_to_beginning(sample_tree):
    """Test seeking to a key at the beginning."""
    store, root_hash, expected_data = sample_tree

    # Seek to key0000
    cursor = TreeCursor(store, root_hash, seek_to=b'key0000')
    results = []

    entry = cursor.next()
    while entry:
        results.append(entry)
        entry = cursor.next()

    # Should get all keys
    assert len(results) == 100
    assert results == expected_data


def test_cursor_seek_to_end(sample_tree):
    """Test seeking past all keys."""
    store, root_hash, expected_data = sample_tree

    # Seek to a key past all existing keys
    cursor = TreeCursor(store, root_hash, seek_to=b'key9999')

    entry = cursor.next()
    assert entry is None


def test_cursor_seek_with_prefix():
    """Test seeking to a prefix."""
    store = MemoryBlockStore()
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)

    # Insert data with different prefixes
    data = []
    for prefix in ['apple', 'banana', 'cherry']:
        for i in range(10):
            data.append((f'{prefix}{i:02d}'.encode(), f'value_{prefix}_{i}'.encode()))

    tree.insert_batch(data, verbose=False)
    root_hash = tree._hash_node(tree.root)

    # Seek to 'banana' prefix
    cursor = TreeCursor(store, root_hash, seek_to=b'banana')
    results = []

    entry = cursor.next()
    while entry and entry[0].startswith(b'banana'):
        results.append(entry)
        entry = cursor.next()

    # Should get all banana keys
    assert len(results) == 10
    assert all(k.startswith(b'banana') for k, v in results)


def test_cursor_empty_tree():
    """Test cursor on empty tree."""
    store = MemoryBlockStore()
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    root_hash = tree._hash_node(tree.root)

    cursor = TreeCursor(store, root_hash)
    entry = cursor.next()

    assert entry is None


def test_cursor_single_item():
    """Test cursor with single item tree."""
    store = MemoryBlockStore()
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(b'key1', b'value1')], verbose=False)
    root_hash = tree._hash_node(tree.root)

    cursor = TreeCursor(store, root_hash)

    entry = cursor.next()
    assert entry == (b'key1', b'value1')

    entry = cursor.next()
    assert entry is None
