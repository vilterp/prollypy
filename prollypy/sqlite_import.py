"""
SQLite database import functionality for ProllyTree.

Provides functions for importing SQLite databases into ProllyTree stores.
"""

import sqlite3
import time
from typing import Optional, List

from .db import DB
from .store import BlockStore, CachedFSBlockStore
from .cursor import TreeCursor


def validate_tree_sorted(db: DB, table_name: str) -> bool:
    """
    Validate that all keys in the tree are in sorted order with no duplicates.

    Args:
        db: DB instance
        table_name: Name of table being validated (for error messages)

    Returns:
        True if valid, False if duplicates or unsorted keys found
    """
    store = db.get_store()
    root_hash = db.get_root_hash()

    cursor = TreeCursor(store, root_hash)
    prev_key = None
    position = 0
    duplicates = []
    unsorted = []

    entry = cursor.next()
    while entry:
        position += 1
        key = entry[0]

        if prev_key is not None:
            if key == prev_key:
                duplicates.append((position, key))
                if len(duplicates) <= 5:  # Show first 5
                    print(f"  DUPLICATE at position {position}: {key}")
            elif key < prev_key:
                unsorted.append((position, prev_key, key))
                if len(unsorted) <= 5:  # Show first 5
                    print(f"  UNSORTED at position {position}: {prev_key} > {key}")

        prev_key = key
        entry = cursor.next()

    if duplicates or unsorted:
        print(f"\nVALIDATION FAILED for {table_name}:")
        print(f"  Total entries: {position:,}")
        print(f"  Duplicates found: {len(duplicates)}")
        print(f"  Unsorted pairs found: {len(unsorted)}")
        return False

    print(f"  Validation: {position:,} keys, all sorted, no duplicates ✓")
    return True


def import_sqlite_table(db: DB, sqlite_conn: sqlite3.Connection, table_name: str,
                        batch_size: int = 1000, verbose_batches: bool = False,
                        validate: bool = False) -> int:
    """
    Import a single SQLite table into the database.

    Args:
        db: DB instance
        sqlite_conn: SQLite connection
        table_name: Name of table to import
        batch_size: Batch size for inserts (0 = insert all rows in one batch)
        verbose_batches: Show detailed batch statistics
        validate: Validate tree is sorted after import

    Returns:
        Number of rows imported
    """
    cursor = sqlite_conn.cursor()

    # Get primary key columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()

    pk_columns = []
    for row in rows:
        if row[5]:  # pk column is at index 5
            pk_columns.append((row[5], row[1]))  # (pk_position, column_name)

    if pk_columns:
        pk_columns.sort(key=lambda x: x[0])
        primary_key = [col[1] for col in pk_columns]
    else:
        primary_key = ["rowid"]

    # Get column names and types
    cursor.execute(f"PRAGMA table_info({table_name})")
    table_info = cursor.fetchall()
    columns = [row[1] for row in table_info]
    column_types = [row[2] for row in table_info]

    # Display table info
    if len(primary_key) == 1:
        print(f"Primary key: {primary_key[0]}")
    else:
        print(f"Primary key (compound): {', '.join(primary_key)}")

    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"Total rows: {row_count:,}")

    if row_count == 0:
        print("Skipping empty table")
        return 0

    # If batch_size is 0, use row_count (import everything in one batch)
    if batch_size == 0:
        batch_size = row_count
        print(f"Using single batch mode: {batch_size:,} rows")

    # Create table in DB
    print(f"Stored schema: {len(columns)} columns")
    db.create_table(table_name, columns, column_types, primary_key)

    # Prepare row iterator - no ORDER BY, we'll sort each batch in Python
    if primary_key == ["rowid"]:
        cursor.execute(f"SELECT rowid, * FROM {table_name}")
    else:
        cursor.execute(f"SELECT * FROM {table_name}")

    def row_generator():
        """Generator that yields rows from cursor."""
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                if primary_key == ["rowid"]:
                    # Skip rowid in the values, it's handled by insert_rows
                    yield row
                else:
                    yield row

    # Insert rows
    table_start = time.time()

    # Enable per-batch validation if requested
    if validate:
        db._validate_after_batch = True

    rows_processed = db.insert_rows(table_name, row_generator(),
                                    batch_size=batch_size,
                                    verbose=verbose_batches)

    # Disable per-batch validation
    if validate:
        db._validate_after_batch = False

    table_time = time.time() - table_start
    table_rate = rows_processed / table_time if table_time > 0 else 0

    print(f"\nCompleted {table_name}: {rows_processed:,} rows in {table_time:.2f}s "
          f"({table_rate:,.0f} rows/sec)")

    # Print root hash
    root_hash = db.get_root_hash()
    print(f"  Root hash: {root_hash.hex()}")

    # Validate if requested
    if validate:
        print(f"\nValidating tree for {table_name}...")
        is_valid = validate_tree_sorted(db, table_name)
        if not is_valid:
            raise ValueError(f"Tree validation failed for {table_name} - contains duplicates or unsorted keys!")

    # Show cache stats if available
    store = db.get_store()
    if isinstance(store, CachedFSBlockStore):
        creation_stats = store.get_creation_stats()
        cache_stats = store.get_cache_stats()
        print(f"  Cumulative: {creation_stats['total_leaves_created']:,} leaves, "
              f"{creation_stats['total_internals_created']:,} internals created")
        print(f"  Cache: {cache_stats['cache_evictions']:,} evictions, "
              f"{cache_stats['cache_size']:,}/{cache_stats['max_cache_size']:,} entries, "
              f"{cache_stats['hit_rate']} hit rate")

        # Print size distributions
        print()
        store.print_distributions(bucket_count=10)

    return rows_processed


def import_sqlite_database(db_path: str, store: BlockStore,
                           pattern: float = 0.01, seed: int = 42,
                           batch_size: int = 1000,
                           tables_filter: Optional[List[str]] = None,
                           verbose_batches: bool = False,
                           validate: bool = False) -> DB:
    """
    Import a SQLite database into ProllyTree.

    Args:
        db_path: Path to SQLite database
        store: BlockStore instance to use
        pattern: ProllyTree split pattern
        seed: Random seed
        batch_size: Batch size for inserts
        tables_filter: Optional list of table names to import
        verbose_batches: Show detailed batch statistics
        validate: Validate tree is sorted after each table import

    Returns:
        DB instance
    """
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [row[0] for row in cursor.fetchall()]

    # Filter tables if requested
    if tables_filter:
        tables = [t for t in all_tables if t in tables_filter]
        print(f"Found {len(all_tables)} tables, importing {len(tables)}: {', '.join(tables)}")
        skipped = [t for t in tables_filter if t not in all_tables]
        if skipped:
            print(f"Warning: Requested tables not found: {', '.join(skipped)}")
    else:
        tables = all_tables
        print(f"Found {len(tables)} tables: {', '.join(tables)}")

    # Create DB
    print(f"\nInitializing ProllyTree (pattern={pattern}, seed={seed})")
    if isinstance(store, CachedFSBlockStore):
        cache_stats = store.get_cache_stats()
        print(f"  Cache size: {cache_stats['max_cache_size']}")

    db = DB(store=store, pattern=pattern, seed=seed, validate=validate)

    # Import each table
    total_rows = 0
    total_start = time.time()

    for table_name in tables:
        print(f"\n{'='*80}")
        print(f"Importing table: {table_name}")
        print(f"{'='*80}")

        rows_imported = import_sqlite_table(db, conn, table_name,
                                            batch_size=batch_size,
                                            verbose_batches=verbose_batches,
                                            validate=validate)
        total_rows += rows_imported

    total_time = time.time() - total_start
    total_rate = total_rows / total_time if total_time > 0 else 0

    # Get final root hash
    final_root_hash = db.get_root_hash()

    print(f"\n{'='*80}")
    print(f"IMPORT COMPLETE")
    print(f"{'='*80}")
    print(f"Total rows imported: {total_rows:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Overall rate: {total_rate:,.0f} rows/sec")
    print(f"\n{'='*80}")
    print(f"FINAL ROOT HASH: {final_root_hash.hex()}")
    print(f"{'='*80}")

    # Print final stats
    print(f"\nTree statistics:")
    print(f"  Store type: {type(store).__name__}")
    print(f"  Total nodes in storage: {store.count_nodes():,}")

    if isinstance(store, CachedFSBlockStore):
        stats = store.get_cache_stats()
        print(f"\nCache statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    conn.close()
    return db
