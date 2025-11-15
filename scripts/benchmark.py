#!/usr/bin/env python3
"""
Benchmark script for ProllyPy - tests insertion performance with random key-value pairs.

This script inserts a configurable number of random key-value pairs into a ProllyTree
and reports detailed timing and performance statistics.
"""

import argparse
import os
import random
import string
import time
from typing import List, Tuple

from prollypy.tree import ProllyTree
from prollypy.store import MemoryStore, FileSystemStore, CachedFSStore


def generate_random_kv_pairs(count: int, key_length: int = 20, value_length: int = 50, seed: int = None) -> List[Tuple[str, str]]:
    """Generate random key-value pairs for benchmarking.

    Args:
        count: Number of key-value pairs to generate
        key_length: Length of each key string
        value_length: Length of each value string
        seed: Random seed for reproducibility

    Returns:
        Sorted list of (key, value) tuples
    """
    if seed is not None:
        random.seed(seed)

    pairs = []
    for i in range(count):
        # Generate random strings with some structure for realistic testing
        key = f"key_{i:010d}_" + ''.join(random.choices(string.ascii_letters + string.digits, k=key_length-15))
        value = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=value_length))
        pairs.append((key, value))

    # ProllyTree requires sorted mutations
    pairs.sort()
    return pairs


def benchmark_insertion(
    num_pairs: int,
    pattern: float = 0.25,
    seed: int = 42,
    store_type: str = "cached",
    key_length: int = 20,
    value_length: int = 10000,
    validate: bool = False,
    base_path: str = "/tmp/prollypy_bench",
    batch_size: int = 1000,
    print_tree: bool = False,
    print_histograms: bool = True,
    sort_globally: bool = False
) -> dict:
    """Run insertion benchmark and return statistics.

    Args:
        num_pairs: Number of key-value pairs to insert
        pattern: Split probability for ProllyTree (lower = larger nodes)
        seed: Seed for rolling hash
        store_type: Type of storage backend ("memory", "filesystem", "cached")
        key_length: Length of each key string
        value_length: Length of each value string
        validate: Whether to enable tree validation (slower)
        base_path: Base path for filesystem storage
        batch_size: Number of rows to insert per batch
        print_tree: Whether to print tree structure at end
        print_histograms: Whether to print node size histograms
        sort_globally: If True, sort all keys globally; if False (default), sort only within batches

    Returns:
        Dictionary containing benchmark results and statistics
    """
    # Create storage backend
    if store_type == "memory":
        store = MemoryStore()
    elif store_type == "filesystem":
        store = FileSystemStore(base_path=base_path)
    elif store_type == "cached":
        store = CachedFSStore(base_path=base_path, cache_size=1000)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

    # Generate random data
    sorting_mode = "globally" if sort_globally else "per-batch"
    print(f"\nGenerating {num_pairs:,} random key-value pairs (sorting: {sorting_mode})...")
    gen_start = time.time()

    if sort_globally:
        # Original behavior: generate and sort all keys globally
        mutations = generate_random_kv_pairs(num_pairs, key_length, value_length, seed=seed + 1000)
        gen_time = time.time() - gen_start
        print(f"  Generated and sorted globally in {gen_time:.2f}s ({num_pairs/gen_time:,.0f} pairs/sec)")
    else:
        # New behavior: generate unsorted, will sort per batch later
        if seed is not None:
            random.seed(seed + 1000)

        mutations = []
        for i in range(num_pairs):
            key = f"key_{i:010d}_" + ''.join(random.choices(string.ascii_letters + string.digits, k=key_length-15))
            value = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=value_length))
            mutations.append((key, value))

        gen_time = time.time() - gen_start
        print(f"  Generated (unsorted) in {gen_time:.2f}s ({num_pairs/gen_time:,.0f} pairs/sec)")

    # Create tree
    print(f"\nCreating ProllyTree (pattern={pattern}, seed={seed}, store={store_type})...")
    tree = ProllyTree(pattern=pattern, seed=seed, store=store, validate=validate)

    # Benchmark insertion in batches
    print(f"\nInserting {num_pairs:,} key-value pairs in batches of {batch_size:,}...")
    insert_start = time.time()

    # Split mutations into batches
    num_batches = (len(mutations) + batch_size - 1) // batch_size
    all_stats = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(mutations))
        batch = mutations[start_idx:end_idx]

        # Sort batch if not globally sorted
        if not sort_globally:
            batch = sorted(batch, key=lambda x: x[0])

        print(f"  Batch {i+1}/{num_batches} ({len(batch):,} rows)... ", end='', flush=True)
        batch_stats = tree.insert_batch(batch, verbose=False)
        all_stats.append(batch_stats)
        print(f"done ({batch_stats.get('nodes_created', 0):,} new nodes)")

    insert_time = time.time() - insert_start

    # Aggregate stats from all batches
    stats = {
        'nodes_created': sum(s.get('nodes_created', 0) for s in all_stats),
        'leaves_created': sum(s.get('leaves_created', 0) for s in all_stats),
        'internals_created': sum(s.get('internals_created', 0) for s in all_stats),
        'nodes_reused': sum(s.get('nodes_reused', 0) for s in all_stats),
    }

    # Calculate throughput
    throughput = num_pairs / insert_time if insert_time > 0 else 0

    # Get root hash
    root_hash = tree._hash_node(tree.root) if tree.root else None

    # Calculate disk usage for filesystem-based stores
    disk_size_bytes = 0
    if store_type in ["filesystem", "cached"]:
        print("\nCalculating disk usage...")
        for subdir in os.listdir(base_path):
            subdir_path = os.path.join(base_path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    if os.path.isfile(file_path):
                        disk_size_bytes += os.path.getsize(file_path)

    # Print histograms if requested
    if print_histograms and store_type in ["filesystem", "cached"]:
        print("\n" + "=" * 70)
        print("NODE SIZE HISTOGRAMS")
        print("=" * 70)
        if isinstance(store, CachedFSStore):
            store.print_distributions(bucket_count=15)
        elif isinstance(store, FileSystemStore):
            store.stats.print_distributions(bucket_count=15)

    # Print tree structure if requested
    if print_tree:
        tree._print_tree(label=f"Final Tree ({num_pairs:,} entries)", verbose=False)

    # Gather results
    results = {
        "num_pairs": num_pairs,
        "pattern": pattern,
        "seed": seed,
        "store_type": store_type,
        "key_length": key_length,
        "value_length": value_length,
        "validate": validate,
        "batch_size": batch_size,
        "sort_globally": sort_globally,
        "generation_time": gen_time,
        "insertion_time": insert_time,
        "total_time": gen_time + insert_time,
        "throughput": throughput,
        "stats": stats,
        "tree_root_hash": root_hash,
        "disk_size_bytes": disk_size_bytes,
    }

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Number of pairs:  {results['num_pairs']:,}")
    print(f"  Batch size:       {results.get('batch_size', 'N/A'):,}")
    sort_mode = "global" if results.get('sort_globally', True) else "per-batch"
    print(f"  Sort mode:        {sort_mode}")
    print(f"  Pattern:          {results['pattern']}")
    print(f"  Seed:             {results['seed']}")
    print(f"  Store type:       {results['store_type']}")
    print(f"  Key length:       {results['key_length']}")
    print(f"  Value length:     {results['value_length']:,}")
    print(f"  Validation:       {results['validate']}")

    print("\nTiming:")
    print(f"  Data generation:  {results['generation_time']:.2f}s")
    print(f"  Insertion:        {results['insertion_time']:.2f}s")
    print(f"  Total:            {results['total_time']:.2f}s")
    print(f"  Throughput:       {results['throughput']:,.0f} pairs/sec")

    print("\nTree Statistics:")
    stats = results['stats']
    if stats:
        print(f"  Nodes created:    {stats.get('nodes_created', 0):,}")
        print(f"  Leaves created:   {stats.get('leaves_created', 0):,}")
        print(f"  Internals created:{stats.get('internals_created', 0):,}")
        print(f"  Nodes reused:     {stats.get('nodes_reused', 0):,}")

    if results.get('tree_root_hash'):
        print(f"  Root hash:        {results['tree_root_hash'][:16]}...")

    # Print disk usage if available
    if results.get('disk_size_bytes', 0) > 0:
        disk_bytes = results['disk_size_bytes']
        disk_mb = disk_bytes / (1024 * 1024)
        disk_gb = disk_bytes / (1024 * 1024 * 1024)
        if disk_gb >= 1:
            print(f"\nDisk Usage:       {disk_gb:.2f} GB ({disk_bytes:,} bytes)")
        else:
            print(f"\nDisk Usage:       {disk_mb:.2f} MB ({disk_bytes:,} bytes)")

    print("=" * 70)


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark ProllyPy insertion performance with random key-value pairs"
    )
    parser.add_argument(
        "-n", "--num-pairs",
        type=int,
        default=500000,
        help="Number of key-value pairs to insert (default: 500000)"
    )
    parser.add_argument(
        "-p", "--pattern",
        type=float,
        default=0.25,
        help="Split probability pattern (default: 0.25, lower = larger nodes)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Seed for rolling hash (default: 42)"
    )
    parser.add_argument(
        "--store",
        type=str,
        choices=["memory", "filesystem", "cached"],
        default="cached",
        help="Storage backend type (default: cached)"
    )
    parser.add_argument(
        "--key-length",
        type=int,
        default=20,
        help="Length of generated keys (default: 20)"
    )
    parser.add_argument(
        "--value-length",
        type=int,
        default=10000,
        help="Length of generated values (default: 10000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to insert per batch (default: 1000)"
    )
    parser.add_argument(
        "--sort-globally",
        action="store_true",
        help="Sort all keys globally before batching (default: sort within each batch only)"
    )
    parser.add_argument(
        "--print-tree",
        action="store_true",
        help="Print tree structure at the end"
    )
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Skip printing node size histograms"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable tree validation (slower)"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/tmp/prollypy_bench",
        help="Base path for filesystem storage (default: /tmp/prollypy_bench)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across different patterns"
    )

    args = parser.parse_args()

    if args.compare:
        # Run multiple benchmarks with different patterns
        patterns = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5]
        print("\nRunning comparison benchmark across different patterns...")
        print(f"Testing with {args.num_pairs:,} pairs each\n")

        all_results = []
        for pattern in patterns:
            results = benchmark_insertion(
                num_pairs=args.num_pairs,
                pattern=pattern,
                seed=args.seed,
                store_type=args.store,
                key_length=args.key_length,
                value_length=args.value_length,
                validate=args.validate,
                base_path=args.base_path,
                batch_size=args.batch_size,
                print_tree=False,  # Don't print tree in comparison mode
                print_histograms=False,  # Don't print histograms in comparison mode
                sort_globally=args.sort_globally
            )
            all_results.append(results)

        # Print comparison summary
        print("\n" + "=" * 90)
        print("COMPARISON SUMMARY")
        print("=" * 90)
        print(f"{'Pattern':<10} {'Throughput':<15} {'Insert Time':<15} {'Nodes':<10} {'Leaves':<10}")
        print("-" * 90)
        for res in all_results:
            print(
                f"{res['pattern']:<10.4f} "
                f"{res['throughput']:>10,.0f} p/s   "
                f"{res['insertion_time']:>10.2f}s      "
                f"{res['stats'].get('nodes_created', 0):>6,}     "
                f"{res['stats'].get('leaves_created', 0):>6,}"
            )
        print("=" * 90)
    else:
        # Run single benchmark
        results = benchmark_insertion(
            num_pairs=args.num_pairs,
            pattern=args.pattern,
            seed=args.seed,
            store_type=args.store,
            key_length=args.key_length,
            value_length=args.value_length,
            validate=args.validate,
            base_path=args.base_path,
            batch_size=args.batch_size,
            print_tree=args.print_tree,
            print_histograms=not args.no_histograms,
            sort_globally=args.sort_globally
        )
        print_results(results)


if __name__ == "__main__":
    main()
