#!/usr/bin/env python3
"""
Test script for split_tables.py - creates mock databases and verifies splitting works.
"""

import sqlite3
import tempfile
import os
from pathlib import Path

def create_test_databases():
    """Create two test databases with a table that has stable and operational columns."""

    # Create temporary databases
    old_db = tempfile.NamedTemporaryFile(suffix='_old.sqlite', delete=False)
    new_db = tempfile.NamedTemporaryFile(suffix='_new.sqlite', delete=False)

    old_db_path = old_db.name
    new_db_path = new_db.name
    old_db.close()
    new_db.close()

    # Create old database
    conn = sqlite3.connect(old_db_path)
    cursor = conn.cursor()

    # Create buses table with stable and operational columns
    cursor.execute('''
        CREATE TABLE buses (
            name TEXT,
            type TEXT,
            owner TEXT,
            va REAL,
            vm REAL,
            zone INTEGER
        )
    ''')

    # Insert 100 rows
    for i in range(100):
        cursor.execute('''
            INSERT INTO buses VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            f'BUS_{i}',
            'TYPE_A' if i % 10 == 0 else 'TYPE_B',
            f'OWNER_{i % 5}',
            100.0 + i,
            1.0,
            i % 3
        ))

    conn.commit()
    conn.close()

    # Create new database with changes
    conn = sqlite3.connect(new_db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE buses (
            name TEXT,
            type TEXT,
            owner TEXT,
            va REAL,
            vm REAL,
            zone INTEGER
        )
    ''')

    # Insert same rows but with different operational values
    for i in range(100):
        cursor.execute('''
            INSERT INTO buses VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            f'BUS_{i}',  # name: unchanged
            'TYPE_A' if i % 10 == 0 else 'TYPE_B',  # type: unchanged
            f'OWNER_{i % 5}',  # owner: unchanged
            100.0 + i + 50,  # va: changed
            1.05,  # vm: changed
            i % 3 if i % 20 != 0 else (i % 3) + 1  # zone: mostly unchanged
        ))

    conn.commit()
    conn.close()

    print(f"Created test databases:")
    print(f"  Old: {old_db_path}")
    print(f"  New: {new_db_path}")
    print()
    print("Expected behavior:")
    print("  - name, type, owner should be stable (0% change)")
    print("  - va, vm should be operational (100% change)")
    print("  - zone should be mostly stable (~5% change)")
    print()

    return old_db_path, new_db_path


def verify_split_database(db_path):
    """Verify the split database has correct schema and data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"Tables in output database: {tables}")

    if 'buses' not in tables:
        print("ERROR: buses table missing!")
        return False

    if 'buses_operational' not in tables:
        print("ERROR: buses_operational table missing!")
        return False

    # Check buses table schema
    cursor.execute("PRAGMA table_info(buses)")
    buses_cols = [row[1] for row in cursor.fetchall()]
    print(f"buses columns: {buses_cols}")

    # Check buses_operational schema
    cursor.execute("PRAGMA table_info(buses_operational)")
    buses_op_cols = [row[1] for row in cursor.fetchall()]
    print(f"buses_operational columns: {buses_op_cols}")

    # Check row counts
    cursor.execute("SELECT COUNT(*) FROM buses")
    buses_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM buses_operational")
    buses_op_count = cursor.fetchone()[0]

    print(f"buses row count: {buses_count}")
    print(f"buses_operational row count: {buses_op_count}")

    if buses_count != buses_op_count:
        print("ERROR: Row counts don't match!")
        return False

    # Check that we can join them
    cursor.execute('''
        SELECT b.rowid, b.name, b.type, b.zone, bo.va, bo.vm
        FROM buses b
        JOIN buses_operational bo ON b.rowid = bo.rowid
        LIMIT 5
    ''')

    print("\nSample joined data:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    return True


if __name__ == '__main__':
    import subprocess

    print("=" * 80)
    print("CREATING TEST DATABASES")
    print("=" * 80)
    print()

    old_db, new_db = create_test_databases()

    # Create output database path
    output_db = tempfile.NamedTemporaryFile(suffix='_split.sqlite', delete=False)
    output_db_path = output_db.name
    output_db.close()

    try:
        print("=" * 80)
        print("RUNNING SPLIT SCRIPT")
        print("=" * 80)
        print()

        # Run the split script
        script_path = Path(__file__).parent / 'split_tables.py'

        # The script now outputs <name>-split.sqlite, so we need to work in a temp dir
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            # Copy databases to temp dir with simple names
            temp_old = os.path.join(temp_dir, 'old.sqlite')
            temp_new = os.path.join(temp_dir, 'new.sqlite')
            shutil.copy(old_db, temp_old)
            shutil.copy(new_db, temp_new)

            # Run the script from temp dir
            result = subprocess.run([
                'uv', 'run', 'python', str(script_path),
                temp_old, temp_new,
                '--threshold', '10.0',
                '--min-skew', '50.0'
            ], capture_output=True, text=True, cwd=temp_dir)

            # Copy output back
            split_old = os.path.join(temp_dir, 'old-split.sqlite')
            if os.path.exists(split_old):
                shutil.copy(split_old, output_db_path)

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            if result.returncode != 0:
                print(f"ERROR: Script failed with exit code {result.returncode}")
            else:
                print()
                print("=" * 80)
                print("VERIFYING SPLIT DATABASE")
                print("=" * 80)
                print()

                if verify_split_database(output_db_path):
                    print()
                    print("✓ Test passed!")
                else:
                    print()
                    print("✗ Test failed!")
        finally:
            shutil.rmtree(temp_dir)

    finally:
        # Clean up
        os.unlink(old_db)
        os.unlink(new_db)
        os.unlink(output_db_path)
        print()
        print("Cleaned up temporary files")
