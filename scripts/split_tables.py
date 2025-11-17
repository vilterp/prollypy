#!/usr/bin/env python3
"""
Split SQLite tables based on column change frequency.

This script analyzes two versions of a database to identify columns that change
frequently vs rarely, then creates a new database schema where tables are split
into stable and operational parts.

Example:
    buses table with columns [id, name, type, va, vm, zone]

    If va/vm change 97%+ but name/type change <1%, split into:
    - buses: [id, name, type]  (stable columns)
    - buses_operational: [id, va, vm, zone]  (frequently-changing columns)
"""

import sqlite3
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prollypy.db import DB
from prollypy.db_diff import diff_db
from prollypy.store import create_store_from_spec
from prollypy.repo import Repo


def analyze_column_changes(old_db_path: str, new_db_path: str,
                           pattern: float = 0.01, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Analyze column change frequencies between two SQLite databases.

    Returns:
        Dict mapping table_name -> {column_name: change_percentage}
    """
    # Import both databases into temporary prolly stores
    print(f"Analyzing {old_db_path} vs {new_db_path}...")

    old_store = create_store_from_spec(":memory:")
    new_store = create_store_from_spec(":memory:")

    # Load old database
    old_db = DB(store=old_store, pattern=pattern, seed=seed)
    conn = sqlite3.connect(old_db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    # Import each table
    for table in tables:
        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        schema_info = cursor.fetchall()
        columns = [row[1] for row in schema_info]
        types = [row[2] for row in schema_info]

        # Create table with rowid as primary key
        old_db.create_table(table, columns, types, ["rowid"])

        # Insert data
        cursor.execute(f"SELECT rowid, * FROM {table}")
        rows = cursor.fetchall()
        old_db.insert_rows(table, iter(rows), batch_size=1000)

    conn.close()

    # Load new database
    new_db = DB(store=new_store, pattern=pattern, seed=seed)
    conn = sqlite3.connect(new_db_path)
    cursor = conn.cursor()

    for table in tables:
        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        schema_info = cursor.fetchall()
        columns = [row[1] for row in schema_info]
        types = [row[2] for row in schema_info]

        # Create table
        new_db.create_table(table, columns, types, ["rowid"])

        # Insert data
        cursor.execute(f"SELECT rowid, * FROM {table}")
        rows = cursor.fetchall()
        new_db.insert_rows(table, iter(rows), batch_size=1000)

    conn.close()

    # Diff the databases
    print("Computing diff...")
    db_diff_result = diff_db(old_db, new_db)

    # Extract column change percentages
    column_stats = {}
    for table_name, table_diff in db_diff_result.modified_tables.items():
        if not table_diff.modified_rows:
            continue

        num_modified = len(table_diff.modified_rows)
        column_stats[table_name] = {}

        # Get all columns from the schema
        schema = table_diff.new_schema or table_diff.old_schema
        if schema:
            # Initialize all columns with 0% change
            for col in schema.columns:
                column_stats[table_name][col] = 0.0

            # Update with actual change counts
            for col, count in table_diff.column_change_counts.items():
                pct = (count / num_modified * 100) if num_modified > 0 else 0
                column_stats[table_name][col] = pct

        # Debug output
        print(f"\n{table_name}: {num_modified} modified rows")
        for col, pct in sorted(column_stats[table_name].items(), key=lambda x: -x[1]):
            print(f"  {col}: {pct:.1f}%")

    return column_stats


def get_primary_key(db_path: str, table_name: str) -> List[str]:
    """
    Get the primary key column(s) for a table.

    Args:
        db_path: Path to SQLite database
        table_name: Name of table

    Returns:
        List of primary key column names
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get table info to find primary key
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    # Extract columns marked as primary key
    pk_cols = [row[1] for row in columns if row[5] > 0]  # row[5] is pk flag

    if not pk_cols:
        # No explicit primary key, check for composite key in schema
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        schema = cursor.fetchone()
        if schema:
            schema_sql = schema[0]
            # Look for PRIMARY KEY (...) clause
            import re
            match = re.search(r'PRIMARY KEY\s*\(([^)]+)\)', schema_sql, re.IGNORECASE)
            if match:
                pk_str = match.group(1)
                pk_cols = [col.strip() for col in pk_str.split(',')]

    conn.close()
    return pk_cols if pk_cols else []


def identify_splits(column_stats: Dict[str, Dict[str, float]],
                    db_path: str,
                    threshold: float = 10.0,
                    min_skew: float = 50.0) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    """
    Identify which tables should be split and how.

    Args:
        column_stats: Table -> column -> change percentage
        db_path: Path to database (for extracting primary keys)
        threshold: Columns changing less than this % are "stable"
        min_skew: Only split if there's at least this much difference between
                  stable and operational columns

    Returns:
        Dict mapping table_name -> (stable_columns, operational_columns, primary_key_columns)
    """
    splits = {}

    for table_name, col_changes in column_stats.items():
        if not col_changes:
            continue

        # Get primary key for this table
        pk_cols = get_primary_key(db_path, table_name)

        # Sort columns by change frequency
        stable_cols = []
        operational_cols = []

        for col, pct in col_changes.items():
            # Primary key columns always go in the stable table
            if col in pk_cols:
                continue  # PK will be added separately

            if pct < threshold:
                stable_cols.append((col, pct))
            else:
                operational_cols.append((col, pct))

        # Only split if we have both types and significant skew
        if stable_cols and operational_cols:
            max_stable_pct = max(pct for _, pct in stable_cols)
            min_operational_pct = min(pct for _, pct in operational_cols)
            skew = min_operational_pct - max_stable_pct

            if skew >= min_skew:
                splits[table_name] = (
                    [col for col, _ in stable_cols],
                    [col for col, _ in operational_cols],
                    pk_cols
                )

                print(f"\n{table_name}:")
                print(f"  Primary key: {', '.join(pk_cols)}")
                print(f"  Stable columns ({len(stable_cols)}): {', '.join(col for col, _ in stable_cols)}")
                print(f"  Operational columns ({len(operational_cols)}): {', '.join(col for col, _ in operational_cols)}")
                print(f"  Skew: {skew:.1f}% (max stable: {max_stable_pct:.1f}%, min operational: {min_operational_pct:.1f}%)")

    return splits


def apply_split_spec(input_db: str, output_db: str,
                     splits: Dict[str, Tuple[List[str], List[str], List[str]]]):
    """
    Apply a split specification to a database.

    Args:
        input_db: Path to input database
        output_db: Path to output database
        splits: Table -> (stable_columns, operational_columns, primary_key_columns)
    """
    # Connect to both databases
    in_conn = sqlite3.connect(input_db)
    out_conn = sqlite3.connect(output_db)

    in_cursor = in_conn.cursor()
    out_cursor = out_conn.cursor()

    # Get all tables
    in_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [row[0] for row in in_cursor.fetchall()]

    for table in all_tables:
        # Get schema
        in_cursor.execute(f"PRAGMA table_info({table})")
        schema_info = in_cursor.fetchall()
        all_columns = [row[1] for row in schema_info]
        column_types = {row[1]: row[2] for row in schema_info}

        if table not in splits:
            # Copy table as-is
            print(f"\nCopying {table} unchanged...")

            # Get the CREATE TABLE statement from original database
            in_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
            create_sql = in_cursor.fetchone()[0]
            out_cursor.execute(create_sql)

            # Copy data
            in_cursor.execute(f"SELECT * FROM {table}")
            rows = in_cursor.fetchall()
            placeholders = ','.join(['?'] * len(all_columns))
            out_cursor.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        else:
            # Split table
            stable_cols, operational_cols, pk_cols = splits[table]

            # If no primary key, use rowid
            use_rowid = len(pk_cols) == 0

            print(f"\nSplitting {table}...")
            if use_rowid:
                print(f"  Base table: rowid + {len(stable_cols)} columns")
                print(f"  Operational table: rowid + {len(operational_cols)} columns")
            else:
                print(f"  Base table: {len(pk_cols)} PK + {len(stable_cols)} columns")
                print(f"  Operational table: {len(pk_cols)} PK + {len(operational_cols)} columns")

            # Create base table with stable columns
            base_col_defs = []

            if use_rowid:
                # Use rowid as primary key
                base_cols = ['rowid'] + stable_cols
                base_col_defs.append("rowid INTEGER PRIMARY KEY")
            else:
                # Use actual primary key
                base_cols = pk_cols + stable_cols
                # Add primary key columns
                for pk_col in pk_cols:
                    base_col_defs.append(f"{pk_col} {column_types[pk_col]}")

            # Add other stable columns
            for col in stable_cols:
                base_col_defs.append(f"{col} {column_types[col]}")

            # Add PRIMARY KEY constraint if composite key
            if not use_rowid and len(pk_cols) > 1:
                base_col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
            elif not use_rowid and len(pk_cols) == 1:
                base_col_defs[0] += " PRIMARY KEY"

            create_base_sql = f"CREATE TABLE {table} ({', '.join(base_col_defs)})"
            out_cursor.execute(create_base_sql)

            # Create operational table
            op_table = f"{table}_operational"
            op_col_defs = []

            if use_rowid:
                # Use rowid as primary key
                op_cols = ['rowid'] + operational_cols
                op_col_defs.append("rowid INTEGER PRIMARY KEY")
            else:
                # Use actual primary key
                op_cols = pk_cols + operational_cols
                # Add primary key columns
                for pk_col in pk_cols:
                    op_col_defs.append(f"{pk_col} {column_types[pk_col]}")

            # Add operational columns
            for col in operational_cols:
                op_col_defs.append(f"{col} {column_types[col]}")

            # Add PRIMARY KEY constraint if composite key
            if not use_rowid and len(pk_cols) > 1:
                op_col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
            elif not use_rowid and len(pk_cols) == 1:
                op_col_defs[0] += " PRIMARY KEY"

            create_op_sql = f"CREATE TABLE {op_table} ({', '.join(op_col_defs)})"
            out_cursor.execute(create_op_sql)

            # Copy data
            if use_rowid:
                # Need to get rowid explicitly
                in_cursor.execute(f"SELECT rowid, * FROM {table}")
                for row in in_cursor.fetchall():
                    # Build row dict
                    row_dict = {'rowid': row[0]}
                    for i, col in enumerate(all_columns):
                        row_dict[col] = row[i + 1]

                    # Insert into base table
                    base_values = [row_dict[col] for col in base_cols]
                    base_placeholders = ','.join(['?'] * len(base_cols))
                    out_cursor.execute(f"INSERT INTO {table} VALUES ({base_placeholders})", base_values)

                    # Insert into operational table
                    op_values = [row_dict[col] for col in op_cols]
                    op_placeholders = ','.join(['?'] * len(op_cols))
                    out_cursor.execute(f"INSERT INTO {op_table} VALUES ({op_placeholders})", op_values)
            else:
                # Use actual primary key
                in_cursor.execute(f"SELECT * FROM {table}")
                for row in in_cursor.fetchall():
                    # Build row dict
                    row_dict = {}
                    for i, col in enumerate(all_columns):
                        row_dict[col] = row[i]

                    # Insert into base table
                    base_values = [row_dict[col] for col in base_cols]
                    base_placeholders = ','.join(['?'] * len(base_cols))
                    out_cursor.execute(f"INSERT INTO {table} VALUES ({base_placeholders})", base_values)

                    # Insert into operational table
                    op_values = [row_dict[col] for col in op_cols]
                    op_placeholders = ','.join(['?'] * len(op_cols))
                    out_cursor.execute(f"INSERT INTO {op_table} VALUES ({op_placeholders})", op_values)

    # Commit and close
    out_conn.commit()
    in_conn.close()
    out_conn.close()

    print(f"\nCreated {output_db}")


def save_split_spec(splits: Dict[str, Tuple[List[str], List[str], List[str]]], output_file: str):
    """Save split specification to a JSON file."""
    spec = {}
    for table_name, (stable_cols, operational_cols, pk_cols) in splits.items():
        spec[table_name] = {
            "primary_key": pk_cols,
            "stable": stable_cols,
            "operational": operational_cols
        }

    with open(output_file, 'w') as f:
        json.dump(spec, f, indent=2)

    print(f"\nSaved split spec to {output_file}")


def load_split_spec(spec_file: str) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    """Load split specification from a JSON file."""
    with open(spec_file, 'r') as f:
        spec = json.load(f)

    splits = {}
    for table_name, cols in spec.items():
        splits[table_name] = (
            cols["stable"],
            cols["operational"],
            cols["primary_key"]
        )

    return splits


def print_split_spec(splits: Dict[str, Tuple[List[str], List[str], List[str]]]):
    """Print the split specification in a readable format."""
    print("\nSplit Specification:")
    print("=" * 80)
    for table_name, (stable_cols, operational_cols, pk_cols) in sorted(splits.items()):
        print(f"\n{table_name}:")
        print(f"  Primary Key: {', '.join(pk_cols)}")
        print(f"  Base table ({table_name}):")
        print(f"    Columns: {', '.join(pk_cols + stable_cols)}")
        print(f"  Operational table ({table_name}_operational):")
        print(f"    Columns: {', '.join(pk_cols + operational_cols)}")


def main():
    parser = argparse.ArgumentParser(
        description='Split SQLite tables based on column change frequency',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze two databases and split both using the same spec
  %(prog)s db1.sqlite db2.sqlite

  # Apply a saved split spec to a database
  %(prog)s --apply-spec split_spec.json input.sqlite
        '''
    )

    # Mode selection
    parser.add_argument('--apply-spec', metavar='SPEC_FILE',
                       help='Apply an existing split spec from JSON file to a single database')

    # Required positional args (different meaning based on mode)
    parser.add_argument('db_a', help='First database (or input database if --apply-spec)')
    parser.add_argument('db_b', nargs='?', help='Second database (required without --apply-spec)')

    # Analysis parameters
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Columns changing less than this percentage are "stable" (default: 10.0)')
    parser.add_argument('--min-skew', type=float, default=50.0,
                       help='Minimum difference between stable and operational columns to trigger split (default: 50.0)')
    parser.add_argument('--pattern', type=float, default=0.01,
                       help='ProllyTree pattern for analysis (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                       help='ProllyTree seed for analysis (default: 42)')

    # Output
    parser.add_argument('--save-spec', metavar='FILE', default='split_spec.json',
                       help='Save split spec to JSON file (default: split_spec.json)')

    args = parser.parse_args()

    if args.apply_spec:
        # Mode: Apply existing spec to a single database
        if args.db_b is not None:
            parser.error("--apply-spec takes only one database argument")

        print("=" * 80)
        print("LOADING SPLIT SPEC")
        print("=" * 80)
        splits = load_split_spec(args.apply_spec)
        print_split_spec(splits)

        # Generate output filename
        db_base = Path(args.db_a).stem
        db_output = f"{db_base}-split.sqlite"

        print("\n" + "=" * 80)
        print("APPLYING SPLIT SPEC")
        print("=" * 80)
        print(f"\nProcessing {args.db_a} -> {db_output}...")
        apply_split_spec(args.db_a, db_output, splits)

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        print(f"Output: {db_output}")

    else:
        # Mode: Analyze two databases and apply spec to both
        if args.db_b is None:
            parser.error("Two databases required without --apply-spec")

        # Analyze column changes
        print("=" * 80)
        print("ANALYZING COLUMN CHANGES")
        print("=" * 80)
        column_stats = analyze_column_changes(args.db_a, args.db_b,
                                              pattern=args.pattern, seed=args.seed)

        # Identify splits
        print("\n" + "=" * 80)
        print("IDENTIFYING TABLE SPLITS (SPLIT SPEC)")
        print("=" * 80)
        splits = identify_splits(column_stats, args.db_a,
                                threshold=args.threshold,
                                min_skew=args.min_skew)

        if not splits:
            print("\nNo tables need splitting based on current thresholds.")
            return

        # Print and save the split spec
        print_split_spec(splits)
        save_split_spec(splits, args.save_spec)

        # Generate output filenames
        db_a_base = Path(args.db_a).stem
        db_b_base = Path(args.db_b).stem
        db_a_output = f"{db_a_base}-split.sqlite"
        db_b_output = f"{db_b_base}-split.sqlite"

        # Apply split spec to both databases
        print("\n" + "=" * 80)
        print("APPLYING SPLIT SPEC TO BOTH DATABASES")
        print("=" * 80)

        print(f"\nProcessing {args.db_a} -> {db_a_output}...")
        apply_split_spec(args.db_a, db_a_output, splits)

        print(f"\nProcessing {args.db_b} -> {db_b_output}...")
        apply_split_spec(args.db_b, db_b_output, splits)

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        print(f"\nSplit {len(splits)} table(s)")
        print(f"Outputs:")
        print(f"  {db_a_output}")
        print(f"  {db_b_output}")
        print(f"  {args.save_spec}")


if __name__ == '__main__':
    main()
