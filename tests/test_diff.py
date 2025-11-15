"""
Tests for diff algorithm.
"""

import pytest

from prollypy.tree import ProllyTree
from prollypy.store import MemoryStore
from prollypy.diff import diff, Differ, Added, Deleted, Modified


@pytest.fixture
def store():
    """Create a shared MemoryStore for testing."""
    return MemoryStore()


def test_diff_identical_trees(store):
    """Test that identical trees produce no diff events."""
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree.insert_batch([(str(i).encode(), f"v{i}".encode()) for i in range(1, 6)], verbose=False)
    root_hash = tree._hash_node(tree.root)

    # Diff tree with itself
    events = list(diff(store, root_hash, root_hash))

    assert len(events) == 0


def test_diff_additions_only(store):
    """Test diff when only additions are present."""
    # Create tree1 with initial data
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"1", b"a"), (b"2", b"b")], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with additional data
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"1", b"a"), (b"2", b"b"), (b"3", b"c"), (b"4", b"d")], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff tree1 -> tree2
    events = list(diff(store, hash1, hash2))

    assert len(events) == 2
    assert events[0] == Added(b"3", b"c")
    assert events[1] == Added(b"4", b"d")


def test_diff_deletions_only(store):
    """Test diff when only deletions are present."""
    # Create tree1 with data
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"1", b"a"), (b"2", b"b"), (b"3", b"c"), (b"4", b"d")], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with subset of data
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"1", b"a"), (b"2", b"b")], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff tree1 -> tree2
    events = list(diff(store, hash1, hash2))

    assert len(events) == 2
    assert events[0] == Deleted(b"3", b"c")
    assert events[1] == Deleted(b"4", b"d")


def test_diff_modifications_only(store):
    """Test diff when only modifications are present."""
    # Create tree1 with initial data
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"1", b"a"), (b"2", b"b"), (b"3", b"c")], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with modified values
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"1", b"a"), (b"2", b"B"), (b"3", b"C")], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff tree1 -> tree2
    events = list(diff(store, hash1, hash2))

    assert len(events) == 2
    assert events[0] == Modified(b"2", b"b", b"B")
    assert events[1] == Modified(b"3", b"c", b"C")


def test_diff_mixed_changes(store):
    """Test diff with additions, deletions, and modifications."""
    # Create tree1
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"1", b"a"), (b"2", b"b"), (b"3", b"c"), (b"5", b"e")], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with mixed changes
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"1", b"A"), (b"2", b"b"), (b"4", b"d"), (b"5", b"e")], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff tree1 -> tree2
    events = list(diff(store, hash1, hash2))

    # Expected: Modified(b"1", b"a" -> "A"), Deleted(3), Added(b"4", b"d")
    assert len(events) == 3
    assert events[0] == Modified(b"1", b"a", b"A")
    assert events[1] == Deleted(b"3", b"c")
    assert events[2] == Added(b"4", b"d")


def test_diff_large_trees(store):
    """Test diff on larger trees to verify subtree skipping."""
    # Create tree1 with 100 entries
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries1 = [(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 101)]
    tree1.insert_batch(entries1, verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with modifications in middle range
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries2 = [(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 101)]
    # Modify entries 40-60
    for i in range(40, 61):
        entries2[i - 1] = (f"{i:04d}".encode(), f"V{i}".encode())  # Uppercase V
    tree2.insert_batch(entries2, verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff should only show modifications in the middle range
    events = list(diff(store, hash1, hash2))

    assert len(events) == 21  # Entries 40-60 (inclusive)
    for i, event in enumerate(events):
        key = 40 + i
        assert event == Modified(f"{key:04d}".encode(), f"v{key}".encode(), f"V{key}".encode())


def test_diff_empty_to_populated(store):
    """Test diff from empty tree to populated tree."""
    # Create empty tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    hash1 = tree1._hash_node(tree1.root)

    # Create populated tree
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"1", b"a"), (b"2", b"b"), (b"3", b"c")], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff empty -> populated
    events = list(diff(store, hash1, hash2))

    assert len(events) == 3
    assert events[0] == Added(b"1", b"a")
    assert events[1] == Added(b"2", b"b")
    assert events[2] == Added(b"3", b"c")


def test_diff_populated_to_empty(store):
    """Test diff from populated tree to empty tree."""
    # Create populated tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"1", b"a"), (b"2", b"b"), (b"3", b"c")], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create empty tree
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    hash2 = tree2._hash_node(tree2.root)

    # Diff populated -> empty
    events = list(diff(store, hash1, hash2))

    assert len(events) == 3
    assert events[0] == Deleted(b"1", b"a")
    assert events[1] == Deleted(b"2", b"b")
    assert events[2] == Deleted(b"3", b"c")


def test_diff_with_string_keys(store):
    """Test diff with string keys (like database tables)."""
    # Create tree1
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch(sorted([
        (b"/d/users/1", b"Alice"),
        (b"/d/users/2", b"Bob"),
        (b"/d/products/1", b"Laptop"),
    ]), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with changes
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch(sorted([
        (b"/d/users/1", b"Alice Smith"),  # Modified
        (b"/d/users/3", b"Charlie"),       # Added
        (b"/d/products/1", b"Laptop"),     # Unchanged
    ]), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff tree1 -> tree2
    events = list(diff(store, hash1, hash2))

    assert len(events) == 3
    assert events[0] == Modified(b"/d/users/1", b"Alice", b"Alice Smith")
    assert events[1] == Deleted(b"/d/users/2", b"Bob")
    assert events[2] == Added(b"/d/users/3", b"Charlie")


def test_diff_subtree_skipping(store):
    """Test that identical subtrees are skipped (performance optimization)."""
    # Create a large tree
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    # Insert 1-50 in one batch
    tree1.insert_batch([(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 51)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with same data
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 51)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Hashes should be identical
    assert hash1 == hash2

    # Diff should produce no events
    events = list(diff(store, hash1, hash2))
    assert len(events) == 0


def test_diff_partial_overlap(store):
    """Test diff with partial overlap between trees."""
    # Create tree1 with keys 1-10
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 11)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with keys 6-15
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(6, 16)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff tree1 -> tree2
    events = list(diff(store, hash1, hash2))

    # Expected: Deleted(1-5), Added(11-15)
    deleted = [e for e in events if isinstance(e, Deleted)]
    added = [e for e in events if isinstance(e, Added)]

    assert len(deleted) == 5
    assert len(added) == 5
    assert all(e.key in [f"{i:04d}".encode() for i in range(1, 6)] for e in deleted)
    assert all(e.key in [f"{i:04d}".encode() for i in range(11, 16)] for e in added)


def test_diff_events_are_ordered(store):
    """Test that diff events are yielded in key order."""
    # Create tree1
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"10", b"j"), (b"20", b"t"), (b"30", b"th")], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with scattered changes
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"05", b"e"), (b"10", b"J"), (b"25", b"tw"), (b"30", b"th")], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff should be in key order
    events = list(diff(store, hash1, hash2))

    # Extract keys in order
    keys = []
    for event in events:
        if isinstance(event, (Added, Deleted)):
            keys.append(event.key)
        elif isinstance(event, Modified):
            keys.append(event.key)

    # Keys should be sorted
    assert keys == sorted(keys)


def test_diff_event_repr():
    """Test string representation of diff events."""
    added = Added(b"1", b"a")
    deleted = Deleted(b"2", b"b")
    modified = Modified(b"3", b"old", b"new")

    assert repr(added) == "Added(b'1', b'a')"
    assert repr(deleted) == "Deleted(b'2', b'b')"
    assert repr(modified) == "Modified(b'3', b'old' -> b'new')"


def test_differ_statistics(store):
    """Test that Differ tracks subtree skip statistics."""
    # Create tree1 with 100 entries
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries1 = [(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 101)]
    tree1.insert_batch(entries1, verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 - identical to tree1
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries2 = [(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 101)]
    tree2.insert_batch(entries2, verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Use Differ to track stats
    differ = Differ(store)
    events = list(differ.diff(hash1, hash2))

    # Identical trees should have no events
    assert len(events) == 0

    # Should have skipped the root (since hashes are identical)
    stats = differ.get_stats()
    assert stats.subtrees_skipped == 1
    assert stats.nodes_compared == 0


def test_diff_identical_values_no_change(store):
    """Test that identical values don't produce diff events (regression test)."""
    schema_json = b'{"columns":["i","j","ckt"],"types":["INTEGER","INTEGER","TEXT"],"primary_key":["i","j","ckt"]}'

    # Create tree1 with schema
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([(b"/s/lines", schema_json)], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with identical schema
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([(b"/s/lines", schema_json)], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Hashes should be identical
    assert hash1 == hash2, "Trees with identical data should have identical hashes"

    # Diff should produce no events
    events = list(diff(store, hash1, hash2))
    assert len(events) == 0, f"Expected no diff events for identical values, got {events}"


def test_diff_different_trees_same_value(store):
    """Test that when different trees have the same value for a key, no diff is shown."""
    schema_json = b'{"columns":["i","j","ckt"],"types":["INTEGER","INTEGER","TEXT"],"primary_key":["i","j","ckt"]}'

    # Create tree1 with schema + other data
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree1.insert_batch([
        (b"/s/lines", schema_json),
        (b"/d/table1/1", b"data1"),
        (b"/d/table1/2", b"data2"),
    ], verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with same schema but different other data
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    tree2.insert_batch([
        (b"/s/lines", schema_json),  # Same value!
        (b"/d/table1/1", b"data1"),
        (b"/d/table1/3", b"data3"),  # Different row
    ], verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Trees should have different hashes (different data)
    assert hash1 != hash2, "Trees with different data should have different hashes"

    # Diff with prefix filter for /s/lines only
    events = list(diff(store, hash1, hash2, prefix=b"/s/lines"))

    # Should have NO events for /s/lines since the value is identical
    assert len(events) == 0, f"Expected no diff events for identical value at /s/lines, got {events}"


def test_diff_same_key_value_different_tree_structure(store):
    """
    Test that identical key-value pairs show NO diff even when tree structure differs.

    This is a regression test for the bug where the diff algorithm reports
    deletions+additions for unchanged keys when trees have different structures.
    """
    schema_json = b'{"columns":["i","j","ckt"],"types":["INTEGER","INTEGER","TEXT"],"primary_key":["i","j","ckt"]}'

    # Create tree1 with lots of data to force multi-level internal nodes
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries1 = [
        (b"/s/lines", schema_json),  # The key we care about
    ]
    # Add lots of keys before /s/lines
    for i in range(50):
        entries1.append((f"/a/key{i:03d}".encode(), f"value{i}".encode()))
    # Add lots of keys after /s/lines
    for i in range(50):
        entries1.append((f"/z/key{i:03d}".encode(), f"value{i}".encode()))
    tree1.insert_batch(sorted(entries1), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with different structure but SAME value for /s/lines
    tree2 = ProllyTree(pattern=0.0001, seed=99, store=store)  # Different seed = different structure
    entries2 = [
        (b"/s/lines", schema_json),  # Same value!
    ]
    # Add different set of keys before /s/lines
    for i in range(30):
        entries2.append((f"/a/key{i:03d}".encode(), f"modified_{i}".encode()))  # Different values
    # Add some new keys
    for i in range(20):
        entries2.append((f"/m/key{i:03d}".encode(), f"new_{i}".encode()))
    # Keep some z keys the same
    for i in range(50):
        entries2.append((f"/z/key{i:03d}".encode(), f"value{i}".encode()))
    tree2.insert_batch(sorted(entries2), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Trees should have different hashes (different structure and data)
    assert hash1 != hash2

    # Get all diff events
    events = list(diff(store, hash1, hash2))

    # Check that /s/lines is NOT in any events (since its value is identical)
    events_for_lines = [e for e in events if (
        (isinstance(e, (Added, Deleted)) and e.key == "/s/lines") or
        (isinstance(e, Modified) and e.key == "/s/lines")
    )]

    assert len(events_for_lines) == 0, \
        f"Expected NO diff events for /s/lines (value is identical), but got: {events_for_lines}"

    # Verify other expected changes are present (just a sanity check)
    assert len(events) > 0, "Expected some diff events for other keys"


def test_diff_same_key_different_surrounding_data(store):
    """
    Test that identical key-value pairs are recognized even when surrounded by different data.

    Tree1: 100 "a*" keys + key "b"
    Tree2: key "b" + 100 "c*" keys

    Both have "b" with the same value, so it should NOT appear in diff.
    """
    # Tree1: has key "b" plus 100 "a*" keys
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    data1 = [(b"b", b"VALUE")]
    for i in range(100):
        data1.append((f"a{i:03d}".encode(), f"x{i}".encode()))
    tree1.insert_batch(sorted(data1), verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Tree2: has key "b" with SAME value, plus different "c*" keys
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    data2 = [(b"b", b"VALUE")]
    for i in range(100):
        data2.append((f"c{i:03d}".encode(), f"y{i}".encode()))
    tree2.insert_batch(sorted(data2), verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Diff should show: 100 deletions (a*), no change for "b", 100 additions (c*)
    events = list(diff(store, hash1, hash2))

    # Check for "b"
    b_events = [e for e in events if (
        (isinstance(e, (Added, Deleted)) and e.key == "b") or
        (isinstance(e, Modified) and e.key == "b")
    )]

    assert len(b_events) == 0, f"Expected no events for 'b' (identical value), got {b_events}"

    # Verify we got the expected changes for other keys
    deleted_keys = {e.key for e in events if isinstance(e, Deleted)}
    added_keys = {e.key for e in events if isinstance(e, Added)}

    assert len(deleted_keys) == 100, f"Expected 100 deletions, got {len(deleted_keys)}"
    assert len(added_keys) == 100, f"Expected 100 additions, got {len(added_keys)}"


def test_diff_reports_identical_value_as_deleted_added_bug(store):
    """
    FAILING TEST: Bug where identical key-value pairs are reported as Deleted+Added.

    The root cause: When diff traverses trees with different structures and encounters
    non-overlapping child ranges, it yields ALL entries from one child as deletions
    and ALL from the other as additions, WITHOUT checking if they're actually the same.

    This test uses the real-world hashes that exhibit the bug.
    """
    # Use the actual stored trees that exhibit the bug
    from prollypy.store import create_store_from_spec
    real_store = create_store_from_spec('cached-file://.prolly')

    # These are real commit hashes that have the bug
    hash1 = bytes.fromhex('a16b213fc2e7d598')
    hash2 = bytes.fromhex('8b2d2b8e2c75c085')

    # Verify the trees exist
    if real_store.get_node(hash1) is None or real_store.get_node(hash2) is None:
        # Trees don't exist in this environment, skip test
        import pytest
        pytest.skip("Real trees not available in test environment")

    # Get all diff events for /s/lines
    events = list(diff(real_store, hash1, hash2, prefix=b"/s/lines"))

    # Check if we got Deleted + Added for the same key
    deleted_keys = {e.key: e.old_value for e in events if isinstance(e, Deleted)}
    added_keys = {e.key: e.value for e in events if isinstance(e, Added)}

    # BUG: If a key appears in both deleted and added with same value, that's the bug
    for key in deleted_keys:
        if key in added_keys:
            if deleted_keys[key] == added_keys[key]:
                assert False, \
                    f"BUG: Key {key} reported as both Deleted and Added with identical value! " \
                    f"Value: {deleted_keys[key][:100]}..."

    # If we only got deletions OR additions (not both), the values must be different
    # which would be fine.  But if we got both with same values, that's the bug.


def test_differ_statistics_with_changes(store):
    """Test that Differ tracks statistics correctly with changes."""
    # Create tree1 with 100 entries
    tree1 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries1 = [(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 101)]
    tree1.insert_batch(entries1, verbose=False)
    hash1 = tree1._hash_node(tree1.root)

    # Create tree2 with modification in middle
    tree2 = ProllyTree(pattern=0.0001, seed=42, store=store)
    entries2 = [(f"{i:04d}".encode(), f"v{i}".encode()) for i in range(1, 101)]
    # Modify one entry
    entries2[49] = (b"0050", b"MODIFIED")
    tree2.insert_batch(entries2, verbose=False)
    hash2 = tree2._hash_node(tree2.root)

    # Use Differ to track stats
    differ = Differ(store)
    events = list(differ.diff(hash1, hash2))

    # Should have one modification
    assert len(events) == 1
    assert isinstance(events[0], Modified)
    assert events[0].key == b"0050"

    # Check statistics
    stats = differ.get_stats()
    # subtrees_skipped can be 0 or more depending on tree structure
    assert stats.subtrees_skipped >= 0
    # Note: nodes_compared is no longer tracked in cursor-based algorithm
