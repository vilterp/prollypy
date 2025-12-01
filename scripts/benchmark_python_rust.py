#!/usr/bin/env python3
"""
Benchmark comparing Python vs Rust ProllyTree implementations.

This script creates a SQLite database, imports it using both Python and Rust
implementations, and displays the results side by side.
"""

import argparse
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path for prollypy imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prollypy.db import DB
from prollypy.store import CachedFSBlockStore, MemoryBlockStore


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    implementation: str
    rows: int
    import_time: float
    rows_per_sec: float
    root_hash: str
    node_count: int


def create_test_database(
    db_path: str,
    num_rows: int,
    num_columns: int = 5,
    value_size: int = 100,
    seed: int = 42
) -> None:
    """
    Create a SQLite database with test data.

    Args:
        db_path: Path to create the SQLite database
        num_rows: Number of rows to create
        num_columns: Number of columns (besides id)
        value_size: Average size of text values
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with multiple columns
    columns = [f"col{i} TEXT" for i in range(num_columns)]
    cursor.execute(f"""
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            {', '.join(columns)}
        )
    """)

    # Insert rows in batches
    batch_size = 10000
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    for batch_start in range(0, num_rows, batch_size):
        batch_end = min(batch_start + batch_size, num_rows)
        rows = []

        for i in range(batch_start, batch_end):
            values = [
                ''.join(random.choices(chars, k=random.randint(value_size // 2, value_size * 2)))
                for _ in range(num_columns)
            ]
            rows.append((i,) + tuple(values))

        placeholders = ', '.join(['?' for _ in range(num_columns + 1)])
        cursor.executemany(
            f"INSERT INTO test_data VALUES ({placeholders})",
            rows
        )
        conn.commit()

    conn.close()


def benchmark_python(
    sqlite_path: str,
    repo_path: str,
    batch_size: int,
    cache_size: int,
    pattern: float = 0.01,
    seed: int = 42,
    use_memory: bool = False
) -> BenchmarkResult:
    """
    Benchmark the Python implementation.

    Args:
        sqlite_path: Path to SQLite database
        repo_path: Path for prolly repository
        batch_size: Batch size for inserts
        cache_size: LRU cache size
        pattern: Tree split pattern
        seed: Random seed
        use_memory: Use MemoryBlockStore instead of filesystem

    Returns:
        BenchmarkResult with timing and statistics
    """
    # Create storage
    if use_memory:
        store = MemoryBlockStore()
    else:
        store_path = os.path.join(repo_path, "blocks")
        os.makedirs(store_path, exist_ok=True)
        store = CachedFSBlockStore(base_path=store_path, cache_size=cache_size)

    # Create DB
    db = DB(store=store, pattern=pattern, seed=seed)

    # Open SQLite
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Get table info
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    total_rows = 0
    start_time = time.time()

    for table_name in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        columns = [row[1] for row in table_info]
        column_types = [row[2] for row in table_info]

        # Get primary key
        pk_columns = [(row[5], row[1]) for row in table_info if row[5] > 0]
        if pk_columns:
            pk_columns.sort(key=lambda x: x[0])
            primary_key = [col[1] for col in pk_columns]
        else:
            primary_key = ["rowid"]

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        if row_count == 0:
            continue

        # Create table in DB
        db.create_table(table_name, columns, column_types, primary_key)

        # Prepare row iterator
        if primary_key == ["rowid"]:
            cursor.execute(f"SELECT rowid, * FROM {table_name}")
        else:
            cursor.execute(f"SELECT * FROM {table_name}")

        # Insert in batches
        def row_generator():
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield row

        rows_processed = db.insert_rows(table_name, row_generator(),
                                        batch_size=batch_size, verbose=False)
        total_rows += rows_processed

    elapsed = time.time() - start_time

    # Get final stats
    root_hash = db.get_root_hash()
    node_count = store.count_nodes()

    conn.close()

    return BenchmarkResult(
        implementation="Python",
        rows=total_rows,
        import_time=elapsed,
        rows_per_sec=total_rows / elapsed if elapsed > 0 else 0,
        root_hash=root_hash.hex(),
        node_count=node_count
    )


def benchmark_rust(
    sqlite_path: str,
    repo_path: str,
    batch_size: int,
    cache_size: int,
    rust_binary: str,
    use_memory: bool = False
) -> BenchmarkResult:
    """
    Benchmark the Rust implementation.

    Args:
        sqlite_path: Path to SQLite database
        repo_path: Path for prolly repository
        batch_size: Batch size for inserts
        cache_size: LRU cache size
        rust_binary: Path to prolly CLI binary
        use_memory: Use in-memory storage

    Returns:
        BenchmarkResult with timing and statistics
    """
    # Initialize the repo first (only needed for non-memory mode)
    if not use_memory:
        init_cmd = [
            rust_binary, "init",
            "--repo", repo_path,
            "--author", "benchmark"
        ]
        subprocess.run(init_cmd, capture_output=True, check=True)

    # Run import and capture output
    import_cmd = [
        rust_binary, "import",
        sqlite_path,
        "--repo", repo_path,
        "--batch-size", str(batch_size),
        "--cache-size", str(cache_size)
    ]
    if use_memory:
        import_cmd.append("--memory")

    start_time = time.time()
    result = subprocess.run(import_cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"Rust import failed:\n{result.stderr}")
        raise RuntimeError(f"Rust import failed: {result.stderr}")

    # Parse output for statistics
    output = result.stdout

    # Extract metrics from output
    total_rows = 0
    root_hash = ""
    node_count = 0
    rust_time = elapsed  # Use wall clock time

    for line in output.split('\n'):
        if 'Total rows:' in line:
            total_rows = int(line.split(':')[1].strip())
        elif 'Root hash:' in line:
            root_hash = line.split(':')[1].strip()
        elif 'Total nodes:' in line:
            node_count = int(line.split(':')[1].strip())
        elif 'Total time:' in line:
            # Parse rust-reported time (e.g., "Total time: 1.23s")
            time_str = line.split(':')[1].strip()
            rust_time = float(time_str.rstrip('s'))

    return BenchmarkResult(
        implementation="Rust",
        rows=total_rows,
        import_time=rust_time,
        rows_per_sec=total_rows / rust_time if rust_time > 0 else 0,
        root_hash=root_hash,
        node_count=node_count
    )


def print_results_table(results: list[BenchmarkResult], title: str = ""):
    """Print benchmark results in a formatted table."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")

    # Header
    print(f"\n{'Implementation':<15} {'Rows':>12} {'Time (s)':>12} {'Rows/sec':>15} {'Nodes':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r.implementation:<15} {r.rows:>12,} {r.import_time:>12.2f} {r.rows_per_sec:>15,.0f} {r.node_count:>12,}")

    # Comparison
    if len(results) >= 2:
        python_result = next((r for r in results if r.implementation == "Python"), None)
        rust_result = next((r for r in results if r.implementation == "Rust"), None)

        if python_result and rust_result:
            print("-" * 70)
            speedup = python_result.import_time / rust_result.import_time if rust_result.import_time > 0 else 0
            print(f"\n  Speedup: Rust is {speedup:.1f}x faster than Python")

            # Note: root hashes differ because Python uses SHA256 for rolling hash
            # while Rust uses xxHash3 for performance. The tree structure is the same.
            print(f"  Root hashes (expected to differ due to different rolling hash algorithms):")
            print(f"    Python (SHA256): {python_result.root_hash[:24]}...")
            print(f"    Rust (xxHash3):  {rust_result.root_hash[:24]}...")

    print()


def run_benchmark(
    num_rows: int,
    batch_size: int = 10000,
    cache_size: int = 10000,
    value_size: int = 100,
    num_columns: int = 5,
    seed: int = 42,
    keep_data: bool = False,
    python_only: bool = False,
    rust_only: bool = False,
    use_memory: bool = False
) -> list[BenchmarkResult]:
    """
    Run the full benchmark.

    Args:
        num_rows: Number of rows to test
        batch_size: Batch size for inserts
        cache_size: LRU cache size
        value_size: Average size of text values
        num_columns: Number of columns per row
        seed: Random seed
        keep_data: Keep temp directories after benchmark
        python_only: Only run Python benchmark
        rust_only: Only run Rust benchmark
        use_memory: Use in-memory storage (Python only, Rust uses tmpfs)

    Returns:
        List of benchmark results
    """
    # Find Rust binary
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    rust_binary = project_root / "target" / "release" / "prolly"

    if not rust_binary.exists() and not python_only:
        print(f"Rust binary not found at {rust_binary}")
        print("Building Rust CLI...")
        result = subprocess.run(
            ["cargo", "build", "--release", "-p", "prolly-cli"],
            cwd=project_root,
            capture_output=True
        )
        if result.returncode != 0:
            print(f"Failed to build Rust CLI:\n{result.stderr.decode()}")
            if not python_only:
                raise RuntimeError("Could not build Rust CLI")

    # Create temp directories - use /dev/shm for memory mode if available (Linux tmpfs)
    if use_memory and os.path.exists("/dev/shm"):
        temp_base = tempfile.mkdtemp(prefix="prolly_bench_", dir="/dev/shm")
    else:
        temp_base = tempfile.mkdtemp(prefix="prolly_bench_")

    sqlite_path = os.path.join(temp_base, "test.db")
    python_repo = os.path.join(temp_base, "python_repo")
    rust_repo = os.path.join(temp_base, "rust_repo")

    results = []

    try:
        # Create test database
        print(f"\nCreating test database with {num_rows:,} rows...")
        print(f"  Columns: {num_columns}")
        print(f"  Value size: ~{value_size} chars")

        db_start = time.time()
        create_test_database(sqlite_path, num_rows, num_columns, value_size, seed)
        db_time = time.time() - db_start

        db_size = os.path.getsize(sqlite_path) / (1024 * 1024)
        print(f"  Database created in {db_time:.2f}s ({db_size:.1f} MB)")

        # Run Python benchmark
        if not rust_only:
            store_type = "MemoryBlockStore" if use_memory else "CachedFSBlockStore"
            print(f"\nRunning Python benchmark...")
            print(f"  Store: {store_type}")
            print(f"  Batch size: {batch_size}")
            if not use_memory:
                print(f"  Cache size: {cache_size}")

            python_result = benchmark_python(
                sqlite_path, python_repo, batch_size, cache_size, use_memory=use_memory
            )
            results.append(python_result)
            print(f"  Completed: {python_result.rows_per_sec:,.0f} rows/sec")

        # Run Rust benchmark
        if not python_only:
            print(f"\nRunning Rust benchmark...")
            if use_memory:
                print(f"  Store: MemoryBlockStore")
            else:
                print(f"  Store: CachedFSBlockStore")
            print(f"  Batch size: {batch_size}")
            if not use_memory:
                print(f"  Cache size: {cache_size}")

            rust_result = benchmark_rust(
                sqlite_path, rust_repo, batch_size, cache_size, str(rust_binary),
                use_memory=use_memory
            )
            results.append(rust_result)
            print(f"  Completed: {rust_result.rows_per_sec:,.0f} rows/sec")

        return results

    finally:
        if not keep_data:
            shutil.rmtree(temp_base, ignore_errors=True)
        else:
            print(f"\nTest data kept at: {temp_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs Rust ProllyTree implementations"
    )
    parser.add_argument(
        "-n", "--num-rows",
        type=int,
        default=50000,
        help="Number of rows to benchmark (default: 50000)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=10000,
        help="Batch size for inserts (default: 10000)"
    )
    parser.add_argument(
        "-c", "--cache-size",
        type=int,
        default=10000,
        help="LRU cache size (default: 10000)"
    )
    parser.add_argument(
        "--value-size",
        type=int,
        default=100,
        help="Average size of text values (default: 100)"
    )
    parser.add_argument(
        "--num-columns",
        type=int,
        default=5,
        help="Number of columns per row (default: 5)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for data generation (default: 42)"
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep temporary directories after benchmark"
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only run Python benchmark"
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Only run Rust benchmark"
    )
    parser.add_argument(
        "--compare-sizes",
        action="store_true",
        help="Run benchmarks at multiple sizes for comparison"
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use in-memory storage (no disk I/O)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  ProllyTree Python vs Rust Benchmark")
    if args.memory:
        print("  (In-Memory Mode - No Disk I/O)")
    print("=" * 80)

    if args.compare_sizes:
        # Run at multiple sizes
        sizes = [10000, 50000, 100000, 250000]
        all_results = []

        for size in sizes:
            print(f"\n{'=' * 80}")
            print(f"  Testing with {size:,} rows")
            print(f"{'=' * 80}")

            results = run_benchmark(
                num_rows=size,
                batch_size=args.batch_size,
                cache_size=args.cache_size,
                value_size=args.value_size,
                num_columns=args.num_columns,
                seed=args.seed,
                keep_data=False,
                python_only=args.python_only,
                rust_only=args.rust_only,
                use_memory=args.memory
            )
            all_results.extend(results)
            print_results_table(results, f"Results: {size:,} rows")

        # Print summary
        print("\n" + "=" * 80)
        print("  SUMMARY: All Benchmark Results")
        print("=" * 80)

        print(f"\n{'Impl':<8} {'Rows':>12} {'Time (s)':>10} {'Rows/sec':>12} {'Nodes':>10}")
        print("-" * 60)

        for r in all_results:
            print(f"{r.implementation:<8} {r.rows:>12,} {r.import_time:>10.2f} {r.rows_per_sec:>12,.0f} {r.node_count:>10,}")

        # Calculate average speedup
        python_results = [r for r in all_results if r.implementation == "Python"]
        rust_results = [r for r in all_results if r.implementation == "Rust"]

        if python_results and rust_results:
            print("\n" + "-" * 60)
            print("\nSpeedup by dataset size:")

            for size in sizes:
                py = next((r for r in python_results if r.rows == size), None)
                rs = next((r for r in rust_results if r.rows == size), None)
                if py and rs:
                    speedup = py.import_time / rs.import_time if rs.import_time > 0 else 0
                    print(f"  {size:>8,} rows: Rust is {speedup:.1f}x faster")
    else:
        # Single run
        results = run_benchmark(
            num_rows=args.num_rows,
            batch_size=args.batch_size,
            cache_size=args.cache_size,
            value_size=args.value_size,
            num_columns=args.num_columns,
            seed=args.seed,
            keep_data=args.keep_data,
            python_only=args.python_only,
            rust_only=args.rust_only,
            use_memory=args.memory
        )
        title = f"Benchmark Results: {args.num_rows:,} rows"
        if args.memory:
            title += " (In-Memory)"
        print_results_table(results, title)


if __name__ == "__main__":
    main()
