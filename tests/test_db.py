"""
Tests for DB abstraction layer.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from prollypy.db import DB, Table
from prollypy.store import MemoryStore, FileSystemStore, CachedFSStore


@pytest.fixture
def db():
    """Create a DB instance with in-memory storage."""
    store = MemoryStore()
    return DB(store=store, pattern=0.0001, seed=42)


@pytest.fixture
def db_with_table(db):
    """Create a DB with a test table already created."""
    db.create_table(
        name="users",
        columns=["id", "name", "email"],
        types=["INTEGER", "TEXT", "TEXT"],
        primary_key=["id"]
    )
    return db


def test_create_table(db):
    """Test creating a table."""
    table = db.create_table(
        name="users",
        columns=["id", "name", "email"],
        types=["INTEGER", "TEXT", "TEXT"],
        primary_key=["id"]
    )

    assert table.name == "users"
    assert table.columns == ["id", "name", "email"]
    assert table.types == ["INTEGER", "TEXT", "TEXT"]
    assert table.primary_key == ["id"]


def test_get_table(db_with_table):
    """Test retrieving a table schema."""
    table = db_with_table.get_table("users")

    assert table is not None
    assert table.name == "users"
    assert table.columns == ["id", "name", "email"]
    assert table.primary_key == ["id"]


def test_get_nonexistent_table(db):
    """Test retrieving a non-existent table."""
    table = db.get_table("nonexistent")
    assert table is None


def test_list_tables(db):
    """Test listing all tables."""
    # Initially empty
    assert db.list_tables() == []

    # Create some tables
    db.create_table("users", ["id", "name"], ["INTEGER", "TEXT"], ["id"])
    db.create_table("products", ["id", "title"], ["INTEGER", "TEXT"], ["id"])

    tables = db.list_tables()
    assert len(tables) == 2
    assert "users" in tables
    assert "products" in tables


def test_insert_rows_single_pk(db_with_table):
    """Test inserting rows with single-column primary key."""
    rows = [
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com"),
        (3, "Charlie", "charlie@example.com"),
    ]

    count = db_with_table.insert_rows("users", iter(rows), batch_size=10, verbose=False)

    assert count == 3


def test_insert_rows_compound_pk(db):
    """Test inserting rows with compound primary key."""
    db.create_table(
        name="orders",
        columns=["customer_id", "order_id", "product", "amount"],
        types=["INTEGER", "INTEGER", "TEXT", "REAL"],
        primary_key=["customer_id", "order_id"]
    )

    rows = [
        (1, 100, "Laptop", 999.99),
        (1, 101, "Mouse", 29.99),
        (2, 100, "Keyboard", 79.99),
    ]

    count = db.insert_rows("orders", iter(rows), batch_size=10, verbose=False)
    assert count == 3


def test_read_rows_with_reconstruction(db_with_table):
    """Test reading rows with object reconstruction."""
    rows = [
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com"),
    ]
    db_with_table.insert_rows("users", iter(rows), verbose=False)

    # Read with reconstruction
    results = list(db_with_table.read_rows("users", reconstruct=True))

    assert len(results) == 2
    key1, row1 = results[0]
    assert key1 == "/d/users/1"
    assert row1 == {"id": 1, "name": "Alice", "email": "alice@example.com"}

    key2, row2 = results[1]
    assert key2 == "/d/users/2"
    assert row2 == {"id": 2, "name": "Bob", "email": "bob@example.com"}


def test_read_rows_without_reconstruction(db_with_table):
    """Test reading rows as raw arrays."""
    rows = [
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com"),
    ]
    db_with_table.insert_rows("users", iter(rows), verbose=False)

    # Read without reconstruction
    results = list(db_with_table.read_rows("users", reconstruct=False))

    assert len(results) == 2
    key1, row1 = results[0]
    assert row1 == [1, "Alice", "alice@example.com"]

    key2, row2 = results[1]
    assert row2 == [2, "Bob", "bob@example.com"]


def test_read_rows_with_prefix(db_with_table):
    """Test reading rows with prefix filtering."""
    rows = [
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com"),
        (10, "Charlie", "charlie@example.com"),
        (11, "David", "david@example.com"),
    ]
    db_with_table.insert_rows("users", iter(rows), verbose=False)

    # Read rows with prefix "1" (should get id=1, 10, 11)
    results = list(db_with_table.read_rows("users", prefix="1", reconstruct=True))

    assert len(results) == 3
    ids = [row["id"] for _, row in results]
    assert ids == [1, 10, 11]


def test_count_rows(db_with_table):
    """Test counting rows in a table."""
    rows = [
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com"),
        (3, "Charlie", "charlie@example.com"),
    ]
    db_with_table.insert_rows("users", iter(rows), verbose=False)

    count = db_with_table.count_rows("users")
    assert count == 3


def test_get_root_hash(db_with_table):
    """Test getting root hash."""
    rows = [(1, "Alice", "alice@example.com")]
    db_with_table.insert_rows("users", iter(rows), verbose=False)

    root_hash = db_with_table.get_root_hash()
    assert isinstance(root_hash, str)
    assert len(root_hash) > 0  # Hash should be non-empty


def test_get_store(db):
    """Test getting the underlying store."""
    store = db.get_store()
    assert isinstance(store, MemoryStore)


def test_insert_rows_invalid_table(db):
    """Test inserting into non-existent table raises error."""
    with pytest.raises(ValueError, match="does not exist"):
        db.insert_rows("nonexistent", iter([(1, "test")]), verbose=False)


def test_batch_insert(db_with_table):
    """Test that large inserts are batched correctly."""
    # Insert more rows than batch size
    rows = [(i, f"User{i}", f"user{i}@example.com") for i in range(1, 101)]

    count = db_with_table.insert_rows("users", iter(rows), batch_size=10, verbose=False)

    assert count == 100

    # Verify all rows are retrievable
    all_rows = list(db_with_table.read_rows("users", reconstruct=False))
    assert len(all_rows) == 100


def test_filesystem_persistence(db_with_table):
    """Test that DB works with filesystem storage."""
    temp_dir = tempfile.mkdtemp()
    try:
        fs_store = FileSystemStore(temp_dir)
        db = DB(store=fs_store, pattern=0.0001, seed=42)

        # Create table and insert data
        db.create_table("users", ["id", "name"], ["INTEGER", "TEXT"], ["id"])
        rows = [(1, "Alice"), (2, "Bob")]
        db.insert_rows("users", iter(rows), verbose=False)

        # Verify nodes were written to disk
        assert fs_store.count_nodes() > 0

        # Get root hash for later loading
        root_hash = db.get_root_hash()
        assert root_hash is not None
    finally:
        shutil.rmtree(temp_dir)


def test_cached_store(db_with_table):
    """Test that DB works with cached filesystem storage."""
    temp_dir = tempfile.mkdtemp()
    try:
        cached_store = CachedFSStore(temp_dir, cache_size=10)
        db = DB(store=cached_store, pattern=0.0001, seed=42)

        # Create table and insert data
        db.create_table("users", ["id", "name"], ["INTEGER", "TEXT"], ["id"])
        rows = [(i, f"User{i}") for i in range(1, 21)]
        db.insert_rows("users", iter(rows), verbose=False)

        # Check cache stats
        stats = cached_store.get_cache_stats()
        assert stats['cache_size'] > 0
        assert stats['max_cache_size'] == 10
    finally:
        shutil.rmtree(temp_dir)


def test_table_serialization():
    """Test Table to_dict and from_dict methods."""
    table = Table(
        name="users",
        columns=["id", "name", "email"],
        types=["INTEGER", "TEXT", "TEXT"],
        primary_key=["id"]
    )

    # Serialize
    data = table.to_dict()
    assert "columns" in data
    assert "types" in data
    assert "primary_key" in data

    # Deserialize
    table2 = Table.from_dict("users", data)
    assert table2.name == table.name
    assert table2.columns == table.columns
    assert table2.types == table.types
    assert table2.primary_key == table.primary_key
