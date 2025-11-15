"""
CLI for ProllyTree database operations.

Provides commands for importing SQLite databases and dumping data.
"""

import argparse
import sqlite3
import time
import os
import getpass
from datetime import datetime
from typing import Optional, List

from .db import DB
from .store import create_store_from_spec, CachedFSBlockStore, BlockStore
from .diff import Differ, Added, Deleted, Modified
from .sqlite_import import import_sqlite_database, import_sqlite_table, validate_tree_sorted
from .commonality import compute_commonality, print_commonality_report
from .store_gc import garbage_collect, find_garbage_nodes, GCStats
from .tree import ProllyTree
from .repo import Repo, SqliteCommitGraphStore


def _get_author() -> str:
    """Get the current OS user as the author."""
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _get_repo(prolly_dir: str = '.prolly', cache_size: int = 1000, author: Optional[str] = None) -> Repo:
    """
    Open a repository from a .prolly directory.

    Args:
        prolly_dir: Path to .prolly directory
        cache_size: Cache size for block store
        author: Default author for commits (defaults to current user)

    Returns:
        Repo instance

    Raises:
        FileNotFoundError: If the .prolly directory doesn't exist
    """
    if not os.path.exists(prolly_dir):
        raise FileNotFoundError(f"Repository not found at {prolly_dir}. Run 'prolly init' first.")

    # Create stores
    blocks_dir = os.path.join(prolly_dir, 'blocks')
    commits_db = os.path.join(prolly_dir, 'commits.db')

    block_store = CachedFSBlockStore(blocks_dir, cache_size=cache_size)
    commit_graph_store = SqliteCommitGraphStore(commits_db)

    return Repo(block_store, commit_graph_store, default_author=author or _get_author())


def init_repo(prolly_dir: str = '.prolly', author: Optional[str] = None):
    """
    Initialize a new prolly repository.

    Args:
        prolly_dir: Path to .prolly directory
        author: Default author for initial commit (defaults to current user)
    """
    if os.path.exists(prolly_dir):
        print(f"Error: Repository already exists at {prolly_dir}")
        return

    # Create directory structure
    os.makedirs(prolly_dir, exist_ok=True)
    blocks_dir = os.path.join(prolly_dir, 'blocks')
    commits_db = os.path.join(prolly_dir, 'commits.db')

    # Create stores
    block_store = CachedFSBlockStore(blocks_dir, cache_size=1000)
    commit_graph_store = SqliteCommitGraphStore(commits_db)

    # Initialize empty repo with initial commit
    author = author or _get_author()
    repo = Repo.init_empty(block_store, commit_graph_store, default_author=author)

    head_commit, ref = repo.get_head()
    print(f"Initialized empty prolly repository in {prolly_dir}")
    print(f"Initial commit: {head_commit.compute_hash().hex()[:8]}")
    print(f"Branch: {ref}")
    print(f"Author: {author}")


def log_commits(prolly_dir: str = '.prolly', ref: Optional[str] = None, max_count: Optional[int] = None):
    """
    Show commit history.

    Args:
        prolly_dir: Path to .prolly directory
        ref: Ref or commit hash to start from (default: HEAD)
        max_count: Maximum number of commits to show
    """
    repo = _get_repo(prolly_dir)

    count = 0
    for commit_hash, commit in repo.log(start_ref=ref, max_count=max_count):
        print(f"commit {commit_hash.hex()}")
        print(f"Author: {commit.author}")
        print(f"Date:   {datetime.fromtimestamp(commit.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n    {commit.message}\n")
        count += 1

    if count == 0:
        print("No commits found")


def dump_database(root_hash: str, store_spec: str = 'cached-file://.prolly',
                  cache_size: Optional[int] = None,
                  prefix: Optional[str] = None):
    """
    Dump keys from the database.

    Args:
        root_hash: Root hash to load
        store_spec: Store specification
        cache_size: Cache size for cached stores
        prefix: Optional key prefix to dump (default: dump all)
    """

    print(f"Opening store: {store_spec}")
    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Count nodes
    node_count = store.count_nodes()
    print(f"Store contains {node_count:,} total nodes")

    if node_count == 0:
        print("Store is empty")
        return

    # Create tree
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)

    print(f"Loading tree from root hash: {root_hash}")
    root_hash_bytes = bytes.fromhex(root_hash)
    root_node = store.get_node(root_hash_bytes)
    if not root_node:
        print(f"Error: Root hash {root_hash} not found in store")
        return
    tree.root = root_node

    # Use prefix if provided, otherwise dump everything
    prefix_str = prefix or ""
    prefix_bytes = prefix_str.encode('utf-8')

    # Generic dump
    print(f"\nKeys with prefix: '{prefix_str}'")
    count = 0
    for key, value in tree.items(prefix_bytes):
        print(f"{key} => {value}")
        count += 1

    print(f"\nTotal: {count:,} keys found")


def diff_trees(old_hash: str, new_hash: str,
                store_spec: str = 'cached-file://.prolly',
                cache_size: Optional[int] = None,
                limit: Optional[int] = None,
                prefix: Optional[str] = None):
    """
    Diff two trees by their root hashes.

    Args:
        old_hash: Root hash of old tree
        new_hash: Root hash of new tree
        store_spec: Store specification
        cache_size: Cache size for cached stores
        limit: Maximum number of diff events to display (None for all)
        prefix: Optional key prefix to filter diff results
    """
    print("="*80)
    print("DIFF: Comparing two trees by hash")
    print("="*80)
    print(f"Old hash: {old_hash}")
    print(f"New hash: {new_hash}")
    print(f"Store:    {store_spec}")
    if prefix:
        print(f"Prefix:   {prefix}")

    if old_hash == new_hash:
        print("\nTrees are identical (same root hash)")
        return

    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Create Differ instance to track statistics
    differ = Differ(store)

    print(f"\nDiff events (old -> new):")
    print("-"*80)

    event_count = 0
    added_count = 0
    deleted_count = 0
    modified_count = 0

    # Convert hex strings to bytes
    old_hash_bytes = bytes.fromhex(old_hash)
    new_hash_bytes = bytes.fromhex(new_hash)
    prefix_bytes = prefix.encode('utf-8') if prefix else None

    for event in differ.diff(old_hash_bytes, new_hash_bytes, prefix=prefix_bytes):
        event_count += 1

        if limit is None or event_count <= limit:
            if isinstance(event, Added):
                print(f"+ {event.key} = {event.value}")
                added_count += 1
            elif isinstance(event, Deleted):
                print(f"- {event.key} = {event.old_value}")
                deleted_count += 1
            elif isinstance(event, Modified):
                print(f"M {event.key}: {event.old_value} -> {event.new_value}")
                modified_count += 1
        else:
            # Just count without printing
            if isinstance(event, Added):
                added_count += 1
            elif isinstance(event, Deleted):
                deleted_count += 1
            elif isinstance(event, Modified):
                modified_count += 1

    print("-"*80)
    print(f"\nDiff Summary:")
    print(f"  Added:    {added_count:,}")
    print(f"  Deleted:  {deleted_count:,}")
    print(f"  Modified: {modified_count:,}")
    print(f"  Total:    {event_count:,}")

    if limit is not None and event_count > limit:
        print(f"\n(showing first {limit}, {event_count - limit:,} more events omitted)")

    # Print diff statistics
    diff_stats = differ.get_stats()
    print(f"\nDiff Algorithm Statistics:")
    print(f"  Subtrees skipped (identical hashes): {diff_stats.subtrees_skipped:,}")
    print(f"  Nodes compared:                      {diff_stats.nodes_compared:,}")

    # Show cache stats if using cached store
    if isinstance(store, CachedFSBlockStore):
        print("\n" + "="*80)
        print("CACHE STATISTICS")
        print("="*80)
        stats = store.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")


def print_tree_structure(root_hash: str, store_spec: str = 'cached-file://.prolly',
                        cache_size: Optional[int] = None,
                        prefix: Optional[str] = None,
                        verbose: bool = False):
    """
    Print the tree structure for a given root hash.

    Args:
        root_hash: Root hash of tree to visualize
        store_spec: Store specification
        cache_size: Cache size for cached stores
        prefix: Optional key prefix to filter tree visualization
        verbose: If True, show all leaf node values. If False, only show first/last keys and count.
    """

    print("="*80)
    print("TREE STRUCTURE")
    print("="*80)
    print(f"Root hash: {root_hash}")
    print(f"Store:     {store_spec}")
    if prefix:
        print(f"Prefix:    {prefix}")
    if not verbose:
        print(f"Mode:      compact (use --verbose to show all leaf values)")

    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Create tree and load from root hash
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    root_hash_bytes = bytes.fromhex(root_hash)
    root_node = store.get_node(root_hash_bytes)

    if not root_node:
        print(f"\nError: Root hash {root_hash} not found in store")
        return
    tree.root = root_node

    # If prefix is specified, we'll filter in the visualization
    # Note: The _print_tree method doesn't natively support prefix filtering,
    # but we can add it as a label and the user can still see the full tree structure
    label = f"root={root_hash[:8]}"
    if prefix:
        label += f", filtering by prefix='{prefix}'"
        print(f"\nNote: Full tree structure shown. Use 'dump' command to see filtered data.")

    tree._print_tree(label=label, verbose=verbose)


def commonality_analysis(left_hash: str, right_hash: str, store_spec: str = 'cached-file://.prolly',
                         cache_size: Optional[int] = None):
    """
    Compute commonality between two tree roots.

    Args:
        left_hash: Root hash of left tree
        right_hash: Root hash of right tree
        store_spec: Store specification
        cache_size: Cache size for cached stores
    """
    print(f"Opening store: {store_spec}")
    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Convert hex strings to bytes
    left_hash_bytes = bytes.fromhex(left_hash)
    right_hash_bytes = bytes.fromhex(right_hash)

    # Compute commonality
    stats = compute_commonality(store, left_hash_bytes, right_hash_bytes)

    # Print report
    print_commonality_report(left_hash, right_hash, stats)


def get_key(root_hash: str, key: str, store_spec: str = 'cached-file://.prolly',
            cache_size: Optional[int] = None):
    """
    Get a value by key from a tree.

    Args:
        root_hash: Root hash of tree to search
        key: Key to look up
        store_spec: Store specification
        cache_size: Cache size for cached stores
    """

    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Load tree with this root
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    root_hash_bytes = bytes.fromhex(root_hash)
    root_node = store.get_node(root_hash_bytes)
    if root_node is None:
        print(f"Error: Root hash {root_hash} not found in store")
        return

    tree.root = root_node

    # Search for the key
    key_bytes = key.encode('utf-8')
    for k, v in tree.items(key_bytes):
        if k == key_bytes:
            print(v)
            return

    print(f"Key '{key}' not found")


def set_key(root_hash: str, key: str, value: str, store_spec: str = 'cached-file://.prolly',
            cache_size: Optional[int] = None):
    """
    Set a key-value pair in a tree, creating a new root.

    Args:
        root_hash: Root hash of tree to modify
        key: Key to set
        value: Value to set
        store_spec: Store specification
        cache_size: Cache size for cached stores
    """

    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Load tree with this root
    tree = ProllyTree(pattern=0.0001, seed=42, store=store)
    root_hash_bytes = bytes.fromhex(root_hash)
    root_node = store.get_node(root_hash_bytes)
    if root_node is None:
        print(f"Error: Root hash {root_hash} not found in store")
        return

    tree.root = root_node

    # Insert the key-value pair
    key_bytes = key.encode('utf-8')
    value_bytes = value.encode('utf-8')
    tree.insert_batch([(key_bytes, value_bytes)], verbose=False)

    # Get new root hash
    new_root_hash = tree._hash_node(tree.root)

    print(f"{'='*80}")
    print(f"SET COMPLETE")
    print(f"{'='*80}")
    print(f"Old root: {root_hash}")
    print(f"New root: {new_root_hash.hex()}")
    print(f"Key:      {key}")
    print(f"Value:    {value}")
    print(f"{'='*80}")


def gc_command(root_hashes: List[str], store_spec: str = 'cached-file://.prolly',
               cache_size: Optional[int] = None, dry_run: bool = False):
    """
    Run garbage collection on the store.

    Args:
        root_hashes: List of root hashes to keep (everything else is garbage)
        store_spec: Store specification string
        cache_size: Optional cache size for cached stores
        dry_run: If True, only show what would be removed without actually removing
    """
    print(f"{'='*80}")
    print(f"GARBAGE COLLECTION")
    print(f"{'='*80}")
    print(f"Store: {store_spec}")
    print(f"Root hashes to keep: {len(root_hashes)}")
    for i, root_hash in enumerate(root_hashes, 1):
        print(f"  {i}. {root_hash}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will remove garbage)'}")
    print()

    # Create store
    store = create_store_from_spec(store_spec, cache_size=cache_size)

    # Run garbage collection
    print("Analyzing store...")
    # Convert hex strings to bytes
    root_set = {bytes.fromhex(h) for h in root_hashes}
    stats = garbage_collect(store, root_set, dry_run=dry_run)

    print()
    print(f"{'='*80}")
    print(f"GARBAGE COLLECTION RESULTS")
    print(f"{'='*80}")
    print(f"Total nodes in store:     {stats.total_nodes:>10,}")
    print(f"Reachable nodes:          {stats.reachable_nodes:>10,}  ({stats.reachable_percent:>5.1f}%)")
    print(f"Garbage nodes:            {stats.garbage_nodes:>10,}  ({stats.garbage_percent:>5.1f}%)")
    print(f"{'='*80}")

    if dry_run:
        print()
        print("DRY RUN: No nodes were removed.")
        print("To actually remove garbage, run without --dry-run")
    else:
        print()
        print(f"SUCCESS: Removed {stats.garbage_nodes:,} garbage nodes from store.")

    # Show cache statistics if available
    if isinstance(store, CachedFSBlockStore):
        print()
        print(f"{'='*80}")
        print(f"CACHE STATISTICS")
        print(f"{'='*80}")
        cache_stats = store.get_cache_stats()
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ProllyTree Database CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Init subcommand
    init_parser = subparsers.add_parser('init', help='Initialize a new prolly repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Initialize repository in current directory
  python cli.py init

  # Initialize with specific author
  python cli.py init --author "John Doe <john@example.com>"
        ''')
    init_parser.add_argument('--dir', default='.prolly',
                        help='Directory for repository (default: .prolly)')
    init_parser.add_argument('--author', default=None,
                        help='Default author for commits (default: current user)')

    # Log subcommand
    log_parser = subparsers.add_parser('log', help='Show commit history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Show all commits from HEAD
  python cli.py log

  # Show last 5 commits
  python cli.py log --max-count 5

  # Show commits from a specific branch
  python cli.py log --ref main

  # Show commits from a specific commit hash
  python cli.py log --ref 1a2b3c4d
        ''')
    log_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    log_parser.add_argument('--ref', default=None,
                        help='Ref or commit hash to start from (default: HEAD)')
    log_parser.add_argument('--max-count', type=int, default=None,
                        help='Maximum number of commits to show')

    # Import SQLite subcommand
    import_parser = subparsers.add_parser('import-sqlite', help='Import SQLite database into ProllyTree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Import database
  python cli.py import-sqlite database.sqlite --store file:///tmp/data

  # Import specific tables
  python cli.py import-sqlite database.sqlite --tables buses generators --store file:///tmp/data

  # Import with caching
  python cli.py import-sqlite database.sqlite --store cached-file:///tmp/data --cache-size 1000
        ''')

    import_parser.add_argument('database', help='Path to SQLite database file')
    import_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    import_parser.add_argument('--pattern', type=float, default=0.0001,
                        help='Split pattern (default: 0.0001)')
    import_parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    import_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')
    import_parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for inserts (default: 1000)')
    import_parser.add_argument('--tables', nargs='+', default=None,
                        help='Specific table names to import')
    import_parser.add_argument('--verbose-batches', action='store_true',
                        help='Show detailed batch statistics')
    import_parser.add_argument('--validate', action='store_true',
                        help='Validate tree is sorted after each table import')

    # Dump subcommand
    dump_parser = subparsers.add_parser('dump', help='Dump data from ProllyTree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dump all data
  python cli.py dump abc123

  # Dump schemas
  python cli.py dump abc123 --prefix /s/

  # Dump table data
  python cli.py dump abc123 --prefix /d/buses

  # Dump with custom store
  python cli.py dump abc123 --prefix /s/ --store file:///tmp/data

  # Dump without reconstruction (raw arrays)
  python cli.py dump abc123 --prefix /d/buses --no-reconstruct
        ''')

    dump_parser.add_argument('root_hash', help='Root hash of tree to dump')
    dump_parser.add_argument('--prefix', type=str, default=None,
                        help='Key prefix to filter dump (default: dump all)')
    dump_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    dump_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')
    dump_parser.add_argument('--no-reconstruct', action='store_true',
                        help='Show raw arrays instead of reconstructed objects')

    # Diff subcommand
    diff_parser = subparsers.add_parser('diff', help='Diff two trees by root hash',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Diff two trees
  python cli.py diff a16b213fc2e7d598 8b2d2b8e2c75c085

  # Diff with custom store
  python cli.py diff old_hash new_hash --store cached-file:///tmp/data

  # Limit output
  python cli.py diff old_hash new_hash --limit 100

  # Filter by key prefix
  python cli.py diff old_hash new_hash --prefix /d/table_name
        ''')

    diff_parser.add_argument('old_hash', help='Root hash of old tree')
    diff_parser.add_argument('new_hash', help='Root hash of new tree')
    diff_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    diff_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')
    diff_parser.add_argument('--limit', type=int, default=None,
                        help='Maximum diff events to display (default: all)')
    diff_parser.add_argument('--prefix', type=str, default=None,
                        help='Key prefix to filter diff results')

    # Print-tree subcommand
    print_tree_parser = subparsers.add_parser('print-tree', help='Print tree structure by root hash',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Print tree structure (compact mode)
  python cli.py print-tree a16b213fc2e7d598

  # Print tree with all leaf values
  python cli.py print-tree a16b213fc2e7d598 --verbose

  # Print tree with custom store
  python cli.py print-tree a16b213fc2e7d598 --store cached-file:///tmp/data

  # Print tree with prefix label
  python cli.py print-tree a16b213fc2e7d598 --prefix /d/buses
        ''')

    print_tree_parser.add_argument('root_hash', help='Root hash of tree to visualize')
    print_tree_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    print_tree_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')
    print_tree_parser.add_argument('--prefix', type=str, default=None,
                        help='Key prefix label for the visualization')
    print_tree_parser.add_argument('--verbose', action='store_true',
                        help='Show all leaf node values (default: only show first/last keys and count)')

    # Commonality subcommand
    commonality_parser = subparsers.add_parser('commonality', help='Compare two tree roots (Venn diagram)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Compare two trees
  python cli.py commonality 1395402e5fd71b2d 3005a56aaed3813a --store cached-file://.prolly

Note: This shows structural node sharing. Trees created by incrementally modifying
one another will share most nodes (e.g., 94%% for a single key change). Trees built
independently from the same data may have 0%% commonality due to different structure.
        ''')
    commonality_parser.add_argument('left_hash', help='Root hash of left tree')
    commonality_parser.add_argument('right_hash', help='Root hash of right tree')
    commonality_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    commonality_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')

    # Get subcommand
    get_parser = subparsers.add_parser('get', help='Get a value by key',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Get a key from a tree
  python cli.py get 1395402e5fd71b2d /d/buses/100001 --store cached-file://.prolly
        ''')
    get_parser.add_argument('root_hash', help='Root hash of tree')
    get_parser.add_argument('key', help='Key to retrieve')
    get_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    get_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')

    # Set subcommand
    set_parser = subparsers.add_parser('set', help='Set a key-value pair (creates new root)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Set a key in a tree
  python cli.py set 1395402e5fd71b2d mykey myvalue --store cached-file://.prolly
        ''')
    set_parser.add_argument('root_hash', help='Root hash of tree to modify')
    set_parser.add_argument('key', help='Key to set')
    set_parser.add_argument('value', help='Value to set')
    set_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    set_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')

    # GC subcommand
    gc_parser = subparsers.add_parser('gc', help='Garbage collect unreachable nodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Remove garbage nodes (default behavior)
  python cli.py gc 65161256c9d66c53 ce615a7ec196650a --store cached-file://.prolly

  # Dry run - show what would be removed without removing
  python cli.py gc 65161256c9d66c53 --dry-run

  # Keep multiple roots
  python cli.py gc root1 root2 root3 --store cached-file://.prolly

Note: Garbage collection removes all nodes not reachable from the specified
root hashes. This is useful for cleaning up old tree versions. Use --dry-run
to preview what will be removed before actually removing it.
        ''')
    gc_parser.add_argument('roots', nargs='+', help='Root hashes to keep (everything else is garbage)')
    gc_parser.add_argument('--store', default='cached-file://.prolly',
                        help='Store spec (default: cached-file://.prolly)')
    gc_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')
    gc_parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be removed without actually removing (default: actually remove)')

    args = parser.parse_args()

    if args.command == 'init':
        init_repo(
            prolly_dir=args.dir,
            author=args.author
        )
    elif args.command == 'log':
        log_commits(
            prolly_dir=args.dir,
            ref=args.ref,
            max_count=args.max_count
        )
    elif args.command == 'import-sqlite':
        store = create_store_from_spec(args.store, cache_size=args.cache_size)
        import_sqlite_database(
            db_path=args.database,
            store=store,
            pattern=args.pattern,
            seed=args.seed,
            batch_size=args.batch_size,
            tables_filter=args.tables,
            verbose_batches=args.verbose_batches,
            validate=args.validate
        )
    elif args.command == 'dump':
        dump_database(
            root_hash=args.root_hash,
            store_spec=args.store,
            cache_size=args.cache_size,
            prefix=args.prefix
        )
    elif args.command == 'diff':
        diff_trees(
            old_hash=args.old_hash,
            new_hash=args.new_hash,
            store_spec=args.store,
            cache_size=args.cache_size,
            limit=args.limit,
            prefix=args.prefix
        )
    elif args.command == 'print-tree':
        print_tree_structure(
            root_hash=args.root_hash,
            store_spec=args.store,
            cache_size=args.cache_size,
            prefix=args.prefix,
            verbose=args.verbose
        )
    elif args.command == 'commonality':
        commonality_analysis(
            left_hash=args.left_hash,
            right_hash=args.right_hash,
            store_spec=args.store,
            cache_size=args.cache_size
        )
    elif args.command == 'get':
        get_key(
            root_hash=args.root_hash,
            key=args.key,
            store_spec=args.store,
            cache_size=args.cache_size
        )
    elif args.command == 'set':
        set_key(
            root_hash=args.root_hash,
            key=args.key,
            value=args.value,
            store_spec=args.store,
            cache_size=args.cache_size
        )
    elif args.command == 'gc':
        gc_command(
            root_hashes=args.roots,
            store_spec=args.store,
            cache_size=args.cache_size,
            dry_run=args.dry_run
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
