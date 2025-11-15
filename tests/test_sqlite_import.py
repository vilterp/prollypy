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
Tests for SQLite database import functionality.

Creates test SQLite databases, imports them, and verifies the contents.
"""

import pytest
import sys
import sqlite3
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import DB
from store import MemoryStore, create_store_from_spec


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    # Create temporary database file
    fd, db_path = tempfile.mkstemp(suffix='.sqlite')
    os.close(fd)

    # Create and populate database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create users table with single primary key
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            age INTEGER
        )
    ''')

    # Insert test data
    users_data = [
        (1, "Alice", "alice@example.com", 30),
        (2, "Bob", "bob@example.com", 25),
        (3, "Charlie", "charlie@example.com", 35),
        (4, "David", "david@example.com", 28),
        (5, "Eve", "eve@example.com", 32),
    ]
    cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?)', users_data)

    # Create orders table with compound primary key
    cursor.execute('''
        CREATE TABLE orders (
            customer_id INTEGER,
            order_id INTEGER,
            product TEXT NOT NULL,
            amount REAL,
            PRIMARY KEY (customer_id, order_id)
        )
    ''')

    # Insert test data
    orders_data = [
        (1, 100, "Laptop", 999.99),
        (1, 101, "Mouse", 29.99),
        (2, 100, "Keyboard", 79.99),
        (2, 101, "Monitor", 299.99),
        (3, 100, "Headphones", 149.99),
    ]
    cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?)', orders_data)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def imported_db(temp_sqlite_db):
    """Import the SQLite database and return DB instance."""
    from cli import import_sqlite_database

    store = MemoryStore()
    db = DB(store=store, pattern=0.0001, seed=42)

    # Import the database
    sqlite_conn = sqlite3.connect(temp_sqlite_db)

    # Get all table names
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    # Import each table using the import_sqlite_table function
    from cli import import_sqlite_table
    for table_name in tables:
        import_sqlite_table(db, sqlite_conn, table_name, batch_size=100, verbose_batches=False)

    sqlite_conn.close()

    return db


def test_import_tables_exist(imported_db):
    """Test that all tables were imported."""
    tables = imported_db.list_tables()

    assert len(tables) == 2
    assert "users" in tables
    assert "orders" in tables


def test_import_users_table_schema(imported_db):
    """Test that users table schema was correctly imported."""
    table = imported_db.get_table("users")

    assert table is not None
    assert table.name == "users"
    assert table.columns == ["id", "name", "email", "age"]
    assert table.types == ["INTEGER", "TEXT", "TEXT", "INTEGER"]
    assert table.primary_key == ["id"]


def test_import_orders_table_schema(imported_db):
    """Test that orders table schema was correctly imported."""
    table = imported_db.get_table("orders")

    assert table is not None
    assert table.name == "orders"
    assert table.columns == ["customer_id", "order_id", "product", "amount"]
    assert table.types == ["INTEGER", "INTEGER", "TEXT", "REAL"]
    assert table.primary_key == ["customer_id", "order_id"]


def test_import_users_row_count(imported_db):
    """Test that correct number of rows were imported."""
    count = imported_db.count_rows("users")
    assert count == 5


def test_import_orders_row_count(imported_db):
    """Test that correct number of rows were imported."""
    count = imported_db.count_rows("orders")
    assert count == 5


def test_import_users_data_integrity(imported_db):
    """Test that user data was correctly imported."""
    rows = list(imported_db.read_rows("users", reconstruct=True))

    # Sort by id for consistent ordering
    rows = sorted(rows, key=lambda x: x[1]["id"])

    assert len(rows) == 5

    # Check first user
    key, user = rows[0]
    assert key == "/d/users/1"
    assert user["id"] == 1
    assert user["name"] == "Alice"
    assert user["email"] == "alice@example.com"
    assert user["age"] == 30

    # Check last user
    key, user = rows[4]
    assert key == "/d/users/5"
    assert user["id"] == 5
    assert user["name"] == "Eve"
    assert user["email"] == "eve@example.com"
    assert user["age"] == 32


def test_import_orders_data_integrity(imported_db):
    """Test that order data was correctly imported."""
    rows = list(imported_db.read_rows("orders", reconstruct=True))

    # Sort by customer_id, then order_id
    rows = sorted(rows, key=lambda x: (x[1]["customer_id"], x[1]["order_id"]))

    assert len(rows) == 5

    # Check first order (compound primary key)
    key, order = rows[0]
    assert key == "/d/orders/1/100"
    assert order["customer_id"] == 1
    assert order["order_id"] == 100
    assert order["product"] == "Laptop"
    assert order["amount"] == 999.99

    # Check last order
    key, order = rows[4]
    assert key == "/d/orders/3/100"
    assert order["customer_id"] == 3
    assert order["order_id"] == 100
    assert order["product"] == "Headphones"
    assert order["amount"] == 149.99


def test_import_prefix_filtering(imported_db):
    """Test that prefix filtering works on imported data."""
    # Get all orders for customer 1
    customer_1_orders = list(imported_db.read_rows("orders", prefix="1/", reconstruct=True))

    assert len(customer_1_orders) == 2

    # Both should be for customer_id=1
    for key, order in customer_1_orders:
        assert order["customer_id"] == 1

    # Check specific orders
    products = sorted([order["product"] for _, order in customer_1_orders])
    assert products == ["Laptop", "Mouse"]


def test_import_raw_data(imported_db):
    """Test reading raw data without reconstruction."""
    rows = list(imported_db.read_rows("users", reconstruct=False))

    assert len(rows) == 5

    # Check first row is an array
    key, values = rows[0]
    assert isinstance(values, list)
    assert len(values) == 4  # id, name, email, age


def test_import_root_hash(imported_db):
    """Test that root hash is generated."""
    root_hash = imported_db.get_root_hash()

    assert isinstance(root_hash, str)
    assert len(root_hash) > 0


def test_import_deterministic(temp_sqlite_db):
    """Test that importing the same database twice produces the same root hash."""
    # Import first time
    store1 = MemoryStore()
    db1 = DB(store=store1, pattern=0.0001, seed=42)

    sqlite_conn1 = sqlite3.connect(temp_sqlite_db)
    from cli import import_sqlite_table
    cursor = sqlite_conn1.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table_name in tables:
        import_sqlite_table(db1, sqlite_conn1, table_name, batch_size=100, verbose_batches=False)
    sqlite_conn1.close()

    hash1 = db1.get_root_hash()

    # Import second time
    store2 = MemoryStore()
    db2 = DB(store=store2, pattern=0.0001, seed=42)

    sqlite_conn2 = sqlite3.connect(temp_sqlite_db)
    cursor = sqlite_conn2.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table_name in tables:
        import_sqlite_table(db2, sqlite_conn2, table_name, batch_size=100, verbose_batches=False)
    sqlite_conn2.close()

    hash2 = db2.get_root_hash()

    # Hashes should be identical
    assert hash1 == hash2


def test_import_empty_table(temp_sqlite_db):
    """Test importing a database with an empty table."""
    # Add an empty table to the database
    conn = sqlite3.connect(temp_sqlite_db)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE empty_table (id INTEGER PRIMARY KEY, data TEXT)')
    conn.commit()
    conn.close()

    # Import
    store = MemoryStore()
    db = DB(store=store, pattern=0.0001, seed=42)

    sqlite_conn = sqlite3.connect(temp_sqlite_db)
    from cli import import_sqlite_table

    rows_imported = import_sqlite_table(db, sqlite_conn, "empty_table", batch_size=100, verbose_batches=False)
    sqlite_conn.close()

    # Empty tables are skipped - no rows imported and no schema stored
    assert rows_imported == 0

    # Table schema should NOT exist (empty tables are skipped)
    table = db.get_table("empty_table")
    assert table is None


def test_import_with_null_values(temp_sqlite_db):
    """Test importing rows with NULL values."""
    # Add a row with NULL values
    conn = sqlite3.connect(temp_sqlite_db)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (id, name, email, age) VALUES (6, "Frank", NULL, NULL)')
    conn.commit()
    conn.close()

    # Import
    store = MemoryStore()
    db = DB(store=store, pattern=0.0001, seed=42)

    sqlite_conn = sqlite3.connect(temp_sqlite_db)
    from cli import import_sqlite_table
    import_sqlite_table(db, sqlite_conn, "users", batch_size=100, verbose_batches=False)
    sqlite_conn.close()

    # Find the row with NULL values
    rows = list(db.read_rows("users", prefix="6", reconstruct=True))

    assert len(rows) == 1
    key, user = rows[0]
    assert user["id"] == 6
    assert user["name"] == "Frank"
    assert user["email"] is None
    assert user["age"] is None


def test_import_large_batch(temp_sqlite_db):
    """Test importing with a larger dataset to verify batching."""
    # Add more rows to users table
    conn = sqlite3.connect(temp_sqlite_db)
    cursor = conn.cursor()

    # Add 100 more users
    for i in range(6, 106):
        cursor.execute('INSERT INTO users VALUES (?, ?, ?, ?)',
                      (i, f"User{i}", f"user{i}@example.com", 20 + (i % 50)))
    conn.commit()
    conn.close()

    # Import with small batch size
    store = MemoryStore()
    db = DB(store=store, pattern=0.0001, seed=42)

    sqlite_conn = sqlite3.connect(temp_sqlite_db)
    from cli import import_sqlite_table
    rows_imported = import_sqlite_table(db, sqlite_conn, "users", batch_size=10, verbose_batches=False)
    sqlite_conn.close()

    # Should have imported all 105 rows (5 original + 100 new)
    assert rows_imported == 105
    assert db.count_rows("users") == 105


def test_full_import_workflow(temp_sqlite_db):
    """Test complete import workflow from SQLite to ProllyTree to verification."""
    from cli import import_sqlite_database

    store = MemoryStore()

    # Use import_sqlite_database function to import entire database
    sqlite_conn = sqlite3.connect(temp_sqlite_db)
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    db = DB(store=store, pattern=0.0001, seed=42)

    from cli import import_sqlite_table
    total_rows = 0
    for table_name in tables:
        rows = import_sqlite_table(db, sqlite_conn, table_name, batch_size=100, verbose_batches=False)
        total_rows += rows

    sqlite_conn.close()

    # Verify total rows
    assert total_rows == 10  # 5 users + 5 orders

    # Verify tables exist
    assert len(db.list_tables()) == 2

    # Verify we can read data
    users = list(db.read_rows("users", reconstruct=True))
    orders = list(db.read_rows("orders", reconstruct=True))

    assert len(users) == 5
    assert len(orders) == 5

    # Verify root hash exists
    root_hash = db.get_root_hash()
    assert root_hash is not None
    assert len(root_hash) > 0
