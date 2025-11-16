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
from .db_diff import diff_db, DBDiff, TableDiff
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
    if head_commit:
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


def import_sqlite_to_repo(db_path: str, prolly_dir: str = '.prolly',
                          pattern: float = 0.01, seed: int = 42,
                          batch_size: int = 1000, tables_filter: Optional[List[str]] = None,
                          verbose_batches: bool = False, validate: bool = False,
                          message: Optional[str] = None):
    """
    Import SQLite database into repo and create a commit.

    Args:
        db_path: Path to SQLite database
        prolly_dir: Path to .prolly directory
        pattern: ProllyTree split pattern
        seed: Random seed
        batch_size: Batch size for inserts
        tables_filter: Optional list of table names to import
        verbose_batches: Show detailed batch statistics
        validate: Validate tree is sorted after each table import
        message: Commit message (default: auto-generated)
    """
    repo = _get_repo(prolly_dir)

    # Import the database
    db = import_sqlite_database(
        db_path=db_path,
        store=repo.block_store,
        pattern=pattern,
        seed=seed,
        batch_size=batch_size,
        tables_filter=tables_filter,
        verbose_batches=verbose_batches,
        validate=validate
    )

    # Get the root hash
    root_hash = db.get_root_hash()

    # Create commit message
    if message is None:
        db_name = os.path.basename(db_path)
        if tables_filter:
            message = f"Import tables {', '.join(tables_filter)} from {db_name}"
        else:
            message = f"Import from {db_name}"

    # Create commit with the pattern and seed used for the import
    commit = repo.commit(root_hash, message, pattern=pattern, seed=seed)

    print(f"\nCommit created: {commit.compute_hash().hex()[:8]}")
    print(f"Message: {message}")
    print(f"Root hash: {root_hash.hex()}")

    head_commit, ref = repo.get_head()
    print(f"Updated {ref} to {commit.compute_hash().hex()[:8]}")


def dump_from_repo(ref: Optional[str] = None, prolly_dir: str = '.prolly',
                   prefix: Optional[str] = None):
    """
    Dump keys from a ref or HEAD.

    Args:
        ref: Ref name or commit hash (default: HEAD)
        prolly_dir: Repository directory
        prefix: Optional key prefix to dump (default: dump all)
    """
    repo = _get_repo(prolly_dir)

    # Resolve ref
    if ref is None:
        head_commit, ref_name = repo.get_head()
        if head_commit is None:
            print("Error: No commits in repository")
            return
        commit = head_commit
        root_hash_bytes = head_commit.tree_root
        print(f"Dumping from HEAD ({ref_name})")
    else:
        commit_hash = repo.resolve_ref(ref)
        if commit_hash is None:
            print(f"Error: Ref '{ref}' not found")
            return
        commit = repo.get_commit(commit_hash)
        if commit is None:
            print(f"Error: Commit not found")
            return
        root_hash_bytes = commit.tree_root
        print(f"Dumping from ref '{ref}'")

    print(f"Tree root hash: {root_hash_bytes.hex()}")

    # Create tree with pattern and seed from commit
    tree = ProllyTree(pattern=commit.pattern, seed=commit.seed, store=repo.block_store)

    root_node = repo.block_store.get_node(root_hash_bytes)
    if not root_node:
        print(f"Error: Root node not found in store")
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


def diff_refs(old_ref: str, new_ref: Optional[str] = None,
              prolly_dir: str = '.prolly',
              limit: Optional[int] = None,
              prefix: Optional[str] = None):
    """
    Diff two refs or commits.

    Args:
        old_ref: First ref/commit
        new_ref: Second ref/commit (default: HEAD)
        prolly_dir: Repository directory
        limit: Maximum number of diff events to display
        prefix: Optional key prefix to filter diff results
    """
    repo = _get_repo(prolly_dir)

    # Resolve old_ref
    old_commit_hash = repo.resolve_ref(old_ref)
    if old_commit_hash is None:
        print(f"Error: Ref '{old_ref}' not found")
        return
    old_commit = repo.get_commit(old_commit_hash)
    if old_commit is None:
        print(f"Error: Commit not found")
        return
    old_hash_bytes = old_commit.tree_root
    old_label = old_ref

    # Resolve new_ref
    if new_ref is None:
        head_commit, ref_name = repo.get_head()
        if head_commit is None:
            print("Error: No commits in repository")
            return
        new_hash_bytes = head_commit.tree_root
        new_label = f"HEAD ({ref_name})"
    else:
        new_commit_hash = repo.resolve_ref(new_ref)
        if new_commit_hash is None:
            print(f"Error: Ref '{new_ref}' not found")
            return
        new_commit = repo.get_commit(new_commit_hash)
        if new_commit is None:
            print(f"Error: Commit not found")
            return
        new_hash_bytes = new_commit.tree_root
        new_label = new_ref

    print("="*80)
    print("DIFF: Comparing two commits")
    print("="*80)
    print(f"Old: {old_label} (tree: {old_hash_bytes.hex()[:16]}...)")
    print(f"New: {new_label} (tree: {new_hash_bytes.hex()[:16]}...)")
    if prefix:
        print(f"Prefix: {prefix}")

    if old_hash_bytes == new_hash_bytes:
        print("\nTrees are identical (same root hash)")
        return

    # Create Differ instance
    differ = Differ(repo.block_store)

    print(f"\nDiff events (old -> new):")
    print("-"*80)

    event_count = 0
    added_count = 0
    deleted_count = 0
    modified_count = 0

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


def db_diff_refs(old_ref: str, new_ref: Optional[str] = None,
                 prolly_dir: str = '.prolly',
                 tables: Optional[List[str]] = None,
                 verbose: bool = False):
    """
    Schema-aware diff of two database commits.

    Args:
        old_ref: First ref/commit
        new_ref: Second ref/commit (default: HEAD)
        prolly_dir: Repository directory
        tables: Optional list of table names to diff
        verbose: If True, show detailed row changes
    """
    repo = _get_repo(prolly_dir)

    # Resolve old_ref
    old_commit_hash = repo.resolve_ref(old_ref)
    if old_commit_hash is None:
        print(f"Error: Ref '{old_ref}' not found")
        return
    old_commit = repo.get_commit(old_commit_hash)
    if old_commit is None:
        print(f"Error: Commit not found")
        return

    # Resolve new_ref
    if new_ref is None:
        new_commit, ref_name = repo.get_head()
        if new_commit is None:
            print("Error: No commits in repository")
            return
        new_label = f"HEAD ({ref_name})"
    else:
        new_commit_hash = repo.resolve_ref(new_ref)
        if new_commit_hash is None:
            print(f"Error: Ref '{new_ref}' not found")
            return
        new_commit = repo.get_commit(new_commit_hash)
        if new_commit is None:
            print(f"Error: Commit not found")
            return
        new_label = new_ref

    print("="*80)
    print("DATABASE DIFF: Schema-Aware Comparison")
    print("="*80)
    print(f"Old: {old_ref}")
    print(f"New: {new_label}")
    if tables:
        print(f"Tables: {', '.join(tables)}")
    print()

    # Create DB instances from commits
    old_db = DB(store=repo.block_store, pattern=old_commit.pattern, seed=old_commit.seed)
    old_tree = ProllyTree(pattern=old_commit.pattern, seed=old_commit.seed, store=repo.block_store)
    old_root = repo.block_store.get_node(old_commit.tree_root)
    if old_root:
        old_tree.root = old_root
        old_db.tree = old_tree

    new_db = DB(store=repo.block_store, pattern=new_commit.pattern, seed=new_commit.seed)
    new_tree = ProllyTree(pattern=new_commit.pattern, seed=new_commit.seed, store=repo.block_store)
    new_root = repo.block_store.get_node(new_commit.tree_root)
    if new_root:
        new_tree.root = new_root
        new_db.tree = new_tree

    # Perform schema-aware diff
    db_diff_result = diff_db(old_db, new_db, tables=tables)

    # Print results
    if not db_diff_result.has_changes:
        print("No changes detected")
        return

    # Added tables
    if db_diff_result.added_tables:
        print("="*80)
        print("ADDED TABLES")
        print("="*80)
        for table_name in db_diff_result.added_tables:
            table = new_db.get_table(table_name)
            row_count = new_db.count_rows(table_name)
            print(f"  + {table_name} ({row_count} rows)")
            if table:
                print(f"    Columns: {', '.join(table.columns)}")
        print()

    # Removed tables
    if db_diff_result.removed_tables:
        print("="*80)
        print("REMOVED TABLES")
        print("="*80)
        for table_name in db_diff_result.removed_tables:
            table = old_db.get_table(table_name)
            row_count = old_db.count_rows(table_name)
            print(f"  - {table_name} ({row_count} rows)")
            if table:
                print(f"    Columns: {', '.join(table.columns)}")
        print()

    # Modified tables
    if db_diff_result.modified_tables:
        print("="*80)
        print("MODIFIED TABLES")
        print("="*80)
        for table_name, table_diff in sorted(db_diff_result.modified_tables.items()):
            print(f"\n{table_name}:")
            print(f"  {table_diff.summary()}")

            # Show column change statistics
            if table_diff.column_change_counts:
                print(f"\n  Column change counts:")
                for line in table_diff.column_stats_summary().split('\n'):
                    print(f"    {line}")

            if verbose and (table_diff.added_rows or table_diff.removed_rows or table_diff.modified_rows):
                if table_diff.added_rows:
                    print(f"\n  Added rows ({len(table_diff.added_rows)}):")
                    for row in table_diff.added_rows[:10]:  # Show first 10
                        print(f"    + PK={row.primary_key}")
                        if row.new_values:
                            for col, val in list(row.new_values.items())[:5]:
                                print(f"        {col} = {val}")
                    if len(table_diff.added_rows) > 10:
                        print(f"    ... and {len(table_diff.added_rows) - 10} more")

                if table_diff.removed_rows:
                    print(f"\n  Removed rows ({len(table_diff.removed_rows)}):")
                    for row in table_diff.removed_rows[:10]:
                        print(f"    - PK={row.primary_key}")
                    if len(table_diff.removed_rows) > 10:
                        print(f"    ... and {len(table_diff.removed_rows) - 10} more")

                if table_diff.modified_rows:
                    print(f"\n  Modified rows ({len(table_diff.modified_rows)}):")
                    for row in table_diff.modified_rows[:10]:
                        print(f"    ~ PK={row.primary_key}")
                        print(f"      Changed columns: {', '.join(sorted(row.changed_columns))}")
                        if verbose:
                            for col in sorted(row.changed_columns):
                                old_val = row.old_values.get(col) if row.old_values else None
                                new_val = row.new_values.get(col) if row.new_values else None
                                print(f"        {col}: {old_val} -> {new_val}")
                    if len(table_diff.modified_rows) > 10:
                        print(f"    ... and {len(table_diff.modified_rows) - 10} more")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Added tables:    {len(db_diff_result.added_tables)}")
    print(f"Removed tables:  {len(db_diff_result.removed_tables)}")
    print(f"Modified tables: {len(db_diff_result.modified_tables)}")

    total_added_rows = sum(len(t.added_rows) for t in db_diff_result.modified_tables.values())
    total_removed_rows = sum(len(t.removed_rows) for t in db_diff_result.modified_tables.values())
    total_modified_rows = sum(len(t.modified_rows) for t in db_diff_result.modified_tables.values())

    print(f"\nRow changes across all tables:")
    print(f"  Added:    {total_added_rows:,}")
    print(f"  Removed:  {total_removed_rows:,}")
    print(f"  Modified: {total_modified_rows:,}")


def print_tree_structure(ref: Optional[str] = None, prolly_dir: str = '.prolly',
                        cache_size: Optional[int] = None,
                        prefix: Optional[str] = None,
                        verbose: bool = False):
    """
    Print the tree structure for a given ref or commit.

    Args:
        ref: Ref name, commit hash, or "HEAD" (default: "HEAD")
        prolly_dir: Repository directory
        cache_size: Cache size for cached stores
        prefix: Optional key prefix to filter tree visualization
        verbose: If True, show all leaf node values. If False, only show first/last keys and count.
    """
    repo = _get_repo(prolly_dir, cache_size=cache_size or 1000)

    # Default to HEAD if no ref provided
    if ref is None:
        ref = "HEAD"

    # Resolve ref to commit
    commit_hash = repo.resolve_ref(ref)
    if commit_hash is None:
        print(f"Error: Ref '{ref}' not found")
        return
    commit = repo.get_commit(commit_hash)
    if commit is None:
        print(f"Error: Commit not found")
        return
    tree_root = commit.tree_root

    print("="*80)
    print("TREE STRUCTURE")
    print("="*80)
    print(f"Ref:       {ref}")
    print(f"Commit:    {commit_hash.hex()[:8]}")
    print(f"Tree root: {tree_root.hex()}")
    if prefix:
        print(f"Prefix:    {prefix}")
    if not verbose:
        print(f"Mode:      compact (use --verbose to show all leaf values)")

    # Create tree with pattern and seed from commit
    tree = ProllyTree(pattern=commit.pattern, seed=commit.seed, store=repo.block_store)
    root_node = repo.block_store.get_node(tree_root)

    if not root_node:
        print(f"\nError: Tree root not found in store")
        return
    tree.root = root_node

    # If prefix is specified, we'll filter in the visualization
    # Note: The _print_tree method doesn't natively support prefix filtering,
    # but we can add it as a label and the user can still see the full tree structure
    label = f"root={tree_root.hex()[:8]}"
    if prefix:
        label += f", filtering by prefix='{prefix}'"
        print(f"\nNote: Full tree structure shown. Use 'dump' command to see filtered data.")

    tree._print_tree(label=label, verbose=verbose)


def commonality_analysis(left_ref: str, right_ref: str, prolly_dir: str = '.prolly',
                         cache_size: Optional[int] = None):
    """
    Compute commonality between two commits or refs.

    Args:
        left_ref: Left ref/commit
        right_ref: Right ref/commit
        prolly_dir: Repository directory
        cache_size: Cache size for cached stores
    """
    repo = _get_repo(prolly_dir, cache_size=cache_size or 1000)

    # Resolve left ref
    left_commit_hash = repo.resolve_ref(left_ref)
    if left_commit_hash is None:
        print(f"Error: Ref '{left_ref}' not found")
        return
    left_commit = repo.get_commit(left_commit_hash)
    if left_commit is None:
        print(f"Error: Commit not found")
        return
    left_tree_root = left_commit.tree_root

    # Resolve right ref
    right_commit_hash = repo.resolve_ref(right_ref)
    if right_commit_hash is None:
        print(f"Error: Ref '{right_ref}' not found")
        return
    right_commit = repo.get_commit(right_commit_hash)
    if right_commit is None:
        print(f"Error: Commit not found")
        return
    right_tree_root = right_commit.tree_root

    print(f"Left ref:  {left_ref} -> commit {left_commit_hash.hex()[:8]}")
    print(f"Right ref: {right_ref} -> commit {right_commit_hash.hex()[:8]}")
    print()

    # Compute commonality
    stats = compute_commonality(repo.block_store, left_tree_root, right_tree_root)

    # Print report (using tree root hashes for display)
    print_commonality_report(left_tree_root.hex(), right_tree_root.hex(), stats)


def get_key_from_repo(key: str, ref: Optional[str] = None, prolly_dir: str = '.prolly',
                      cache_size: int = 1000, author: Optional[str] = None):
    """
    Get a value by key from a repository ref or HEAD.

    Args:
        key: Key to look up
        ref: Ref name or commit hash (default: HEAD)
        prolly_dir: Repository directory
        cache_size: Cache size for cached stores
        author: Optional author (unused, for compatibility with _get_repo)
    """
    repo = _get_repo(prolly_dir, cache_size=cache_size, author=author)

    # Determine which commit to read from
    if ref is None:
        head_commit, ref_name = repo.get_head()
        if head_commit is None:
            print("Error: No commits in repository")
            return
        tree_root = head_commit.tree_root
        ref_display = f"HEAD ({ref_name})"
    else:
        commit_hash = repo.resolve_ref(ref)
        if commit_hash is None:
            print(f"Error: Ref '{ref}' not found")
            return
        commit = repo.get_commit(commit_hash)
        if commit is None:
            print(f"Error: Commit not found")
            return
        tree_root = commit.tree_root
        ref_display = ref

    # Load tree with pattern and seed from commit
    tree = ProllyTree(pattern=commit.pattern, seed=commit.seed, store=repo.block_store)
    root_node = repo.block_store.get_node(tree_root)
    if root_node is None:
        print(f"Error: Tree root not found in store")
        return

    tree.root = root_node

    # Search for the key
    key_bytes = key.encode('utf-8')
    for k, v in tree.items(key_bytes):
        if k == key_bytes:
            print(f"{'='*80}")
            print(f"GET from {ref_display}")
            print(f"{'='*80}")
            print(f"Key:   {key}")
            print(f"Value: {v.decode('utf-8', errors='replace')}")
            print(f"{'='*80}")
            return

    print(f"Key '{key}' not found")


def set_key_in_repo(key: str, value: str, message: Optional[str] = None,
                    prolly_dir: str = '.prolly', cache_size: int = 1000,
                    author: Optional[str] = None):
    """
    Set a key-value pair in the repository and create a new commit.

    Args:
        key: Key to set
        value: Value to set
        message: Commit message (default: auto-generated)
        prolly_dir: Repository directory
        cache_size: Cache size for cached stores
        author: Optional author (default: current user)
    """
    repo = _get_repo(prolly_dir, cache_size=cache_size, author=author)

    # Get current HEAD
    head_commit, ref_name = repo.get_head()
    if head_commit is None:
        print("Error: No commits in repository. Use 'init' first.")
        return

    # Load current tree with pattern and seed from commit
    tree = ProllyTree(pattern=head_commit.pattern, seed=head_commit.seed, store=repo.block_store)
    root_node = repo.block_store.get_node(head_commit.tree_root)
    if root_node is None:
        print(f"Error: Tree root not found in store")
        return

    tree.root = root_node

    # Insert the key-value pair
    key_bytes = key.encode('utf-8')
    value_bytes = value.encode('utf-8')
    tree.insert_batch([(key_bytes, value_bytes)], verbose=False)

    # Get new root hash
    new_root_hash = tree._hash_node(tree.root)

    # Create commit
    if message is None:
        message = f"Set {key} = {value}"

    commit = repo.commit(new_root_hash, message, author=author)
    commit_hash = commit.compute_hash()

    print(f"{'='*80}")
    print(f"SET COMPLETE")
    print(f"{'='*80}")
    print(f"Key:      {key}")
    print(f"Value:    {value}")
    print(f"Commit:   {commit_hash.hex()[:8]}")
    print(f"Message:  {message}")
    print(f"Branch:   {ref_name}")
    print(f"{'='*80}")


def gc_repo(prolly_dir: str = '.prolly', cache_size: int = 1000,
            dry_run: bool = False, author: Optional[str] = None):
    """
    Run garbage collection on the repository.

    This uses the repository's commit graph to determine which tree roots
    are reachable from any ref, and removes unreachable nodes from the store.

    Args:
        prolly_dir: Repository directory
        cache_size: Cache size for cached stores
        dry_run: If True, only show what would be removed without actually removing
        author: Optional author (unused, for compatibility with _get_repo)
    """
    print(f"{'='*80}")
    print(f"GARBAGE COLLECTION")
    print(f"{'='*80}")
    print(f"Repository: {prolly_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will remove garbage)'}")
    print()

    # Open repository
    repo = _get_repo(prolly_dir, cache_size=cache_size, author=author)

    # Get all reachable tree roots from refs
    print("Computing reachable tree roots from refs...")
    tree_roots = repo.get_reachable_tree_roots()

    # Show refs and their commits
    refs = repo.commit_graph_store.list_refs()
    print(f"\nRefs ({len(refs)}):")
    for ref_name, commit_hash in refs.items():
        commit = repo.commit_graph_store.get_commit(commit_hash)
        if commit:
            tree_root_short = commit.tree_root.hex()[:16]
            print(f"  {ref_name}: {commit_hash.hex()[:8]} (tree: {tree_root_short})")

    print(f"\nReachable tree roots: {len(tree_roots)}")
    for i, tree_root in enumerate(sorted(tree_roots), 1):
        if i <= 10:  # Show first 10
            print(f"  {i}. {tree_root.hex()[:16]}...")
    if len(tree_roots) > 10:
        print(f"  ... and {len(tree_roots) - 10} more")

    # Run garbage collection
    print("\nAnalyzing store...")
    stats = garbage_collect(repo.block_store, tree_roots, dry_run=dry_run)

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
    if isinstance(repo.block_store, CachedFSBlockStore):
        print()
        print(f"{'='*80}")
        print(f"CACHE STATISTICS")
        print(f"{'='*80}")
        cache_stats = repo.block_store.get_cache_stats()
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
    import_parser = subparsers.add_parser('import-sqlite', help='Import SQLite database and commit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Import database
  python cli.py import-sqlite database.sqlite

  # Import specific tables
  python cli.py import-sqlite database.sqlite --tables buses generators

  # Import with custom message
  python cli.py import-sqlite database.sqlite --message "Import production data"
        ''')

    import_parser.add_argument('database', help='Path to SQLite database file')
    import_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    import_parser.add_argument('--pattern', type=float, default=0.01,
                        help='Split pattern - higher = smaller nodes, wider trees (default: 0.01)')
    import_parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    import_parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for inserts (default: 1000)')
    import_parser.add_argument('--tables', nargs='+', default=None,
                        help='Specific table names to import')
    import_parser.add_argument('--message', default=None,
                        help='Commit message (default: auto-generated)')
    import_parser.add_argument('--verbose-batches', action='store_true',
                        help='Show detailed batch statistics')
    import_parser.add_argument('--validate', action='store_true',
                        help='Validate tree is sorted after each table import')

    # Dump subcommand
    dump_parser = subparsers.add_parser('dump', help='Dump data from a ref or HEAD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dump all data from HEAD
  python cli.py dump

  # Dump from a specific branch
  python cli.py dump --ref main

  # Dump schemas from HEAD
  python cli.py dump --prefix /s/

  # Dump table data from a commit
  python cli.py dump --ref abc123 --prefix /d/buses
        ''')

    dump_parser.add_argument('--ref', default=None,
                        help='Ref name or commit hash (default: HEAD)')
    dump_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    dump_parser.add_argument('--prefix', type=str, default=None,
                        help='Key prefix to filter dump (default: dump all)')

    # Diff subcommand
    diff_parser = subparsers.add_parser('diff', help='Diff two commits or refs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Diff a ref with HEAD
  python cli.py diff main

  # Diff two refs
  python cli.py diff main develop

  # Diff two commits
  python cli.py diff abc123 def456

  # Limit output
  python cli.py diff main --limit 100

  # Filter by key prefix
  python cli.py diff main develop --prefix /d/table_name
        ''')

    diff_parser.add_argument('old_ref', help='Old ref/commit')
    diff_parser.add_argument('new_ref', nargs='?', default=None,
                        help='New ref/commit (default: HEAD)')
    diff_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    diff_parser.add_argument('--limit', type=int, default=None,
                        help='Maximum diff events to display (default: all)')
    diff_parser.add_argument('--prefix', type=str, default=None,
                        help='Key prefix to filter diff results')

    # DB-diff subcommand (schema-aware diff)
    db_diff_parser = subparsers.add_parser('db-diff', help='Schema-aware database diff',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Diff current HEAD with previous commit
  python cli.py db-diff HEAD~1

  # Diff two refs
  python cli.py db-diff main develop

  # Diff specific tables only
  python cli.py db-diff main --tables buses generators

  # Show detailed row changes
  python cli.py db-diff HEAD~1 --verbose
        ''')

    db_diff_parser.add_argument('old_ref', help='Old ref/commit')
    db_diff_parser.add_argument('new_ref', nargs='?', default=None,
                        help='New ref/commit (default: HEAD)')
    db_diff_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    db_diff_parser.add_argument('--tables', nargs='+', default=None,
                        help='Specific tables to diff (default: all tables)')
    db_diff_parser.add_argument('--verbose', action='store_true',
                        help='Show detailed row changes (default: summary only)')

    # Print-tree subcommand
    print_tree_parser = subparsers.add_parser('print-tree', help='Print tree structure for a ref/commit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Print tree structure from HEAD (compact mode)
  python cli.py print-tree

  # Print tree from a branch
  python cli.py print-tree main

  # Print tree from a commit hash
  python cli.py print-tree a16b213f --verbose

  # Print tree with prefix label
  python cli.py print-tree --prefix /d/buses
        ''')

    print_tree_parser.add_argument('ref', nargs='?', default=None, help='Ref name, commit hash, or "HEAD" (default: HEAD)')
    print_tree_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    print_tree_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')
    print_tree_parser.add_argument('--prefix', type=str, default=None,
                        help='Key prefix label for the visualization')
    print_tree_parser.add_argument('--verbose', action='store_true',
                        help='Show all leaf node values (default: only show first/last keys and count)')

    # Commonality subcommand
    commonality_parser = subparsers.add_parser('commonality', help='Compare two commits/refs (Venn diagram)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Compare two commits
  python cli.py commonality abc123 def456

  # Compare two refs
  python cli.py commonality main develop

  # Compare HEAD with a ref
  python cli.py commonality HEAD main

Note: This shows structural node sharing. Trees created by incrementally modifying
one another will share most nodes (e.g., 94%% for a single key change). Trees built
independently from the same data may have 0%% commonality due to different structure.
        ''')
    commonality_parser.add_argument('left_ref', help='Left ref/commit')
    commonality_parser.add_argument('right_ref', help='Right ref/commit')
    commonality_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    commonality_parser.add_argument('--cache-size', type=int, default=None,
                        help='Cache size for cached stores')

    # Get subcommand
    get_parser = subparsers.add_parser('get', help='Get a value by key from repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Get a key from HEAD
  python cli.py get mykey

  # Get a key from a specific ref
  python cli.py get mykey --ref main

  # Get a key from a specific commit
  python cli.py get /d/users/1 --ref abc123
        ''')
    get_parser.add_argument('key', help='Key to retrieve')
    get_parser.add_argument('--ref', default=None,
                        help='Ref name or commit hash (default: HEAD)')
    get_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    get_parser.add_argument('--cache-size', type=int, default=1000,
                        help='Cache size for cached stores (default: 1000)')

    # Set subcommand
    set_parser = subparsers.add_parser('set', help='Set a key-value pair and commit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Set a key on HEAD
  python cli.py set mykey myvalue

  # Set with custom commit message
  python cli.py set mykey myvalue --message "Update mykey value"

  # Set using custom author
  python cli.py set mykey myvalue --author "Alice <alice@example.com>"
        ''')
    set_parser.add_argument('key', help='Key to set')
    set_parser.add_argument('value', help='Value to set')
    set_parser.add_argument('--message', default=None,
                        help='Commit message (default: auto-generated)')
    set_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    set_parser.add_argument('--cache-size', type=int, default=1000,
                        help='Cache size for cached stores (default: 1000)')
    set_parser.add_argument('--author', default=None,
                        help='Commit author (default: current user)')

    # GC subcommand
    gc_parser = subparsers.add_parser('gc', help='Garbage collect unreachable nodes from repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dry run - show what would be removed without removing
  python cli.py gc --dry-run

  # Remove garbage nodes (live mode)
  python cli.py gc

  # Use custom repository directory
  python cli.py gc --dir /path/to/repo

Note: Garbage collection automatically determines which tree nodes are reachable
from any ref in the repository, and removes all unreachable nodes. Use --dry-run
to preview what will be removed before actually removing it.
        ''')
    gc_parser.add_argument('--dir', default='.prolly',
                        help='Repository directory (default: .prolly)')
    gc_parser.add_argument('--cache-size', type=int, default=1000,
                        help='Cache size for cached stores (default: 1000)')
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
        import_sqlite_to_repo(
            db_path=args.database,
            prolly_dir=args.dir,
            pattern=args.pattern,
            seed=args.seed,
            batch_size=args.batch_size,
            tables_filter=args.tables,
            verbose_batches=args.verbose_batches,
            validate=args.validate,
            message=args.message
        )
    elif args.command == 'dump':
        dump_from_repo(
            ref=args.ref,
            prolly_dir=args.dir,
            prefix=args.prefix
        )
    elif args.command == 'diff':
        diff_refs(
            old_ref=args.old_ref,
            new_ref=args.new_ref,
            prolly_dir=args.dir,
            limit=args.limit,
            prefix=args.prefix
        )
    elif args.command == 'db-diff':
        db_diff_refs(
            old_ref=args.old_ref,
            new_ref=args.new_ref,
            prolly_dir=args.dir,
            tables=args.tables,
            verbose=args.verbose
        )
    elif args.command == 'print-tree':
        print_tree_structure(
            ref=args.ref,
            prolly_dir=args.dir,
            cache_size=args.cache_size,
            prefix=args.prefix,
            verbose=args.verbose
        )
    elif args.command == 'commonality':
        commonality_analysis(
            left_ref=args.left_ref,
            right_ref=args.right_ref,
            prolly_dir=args.dir,
            cache_size=args.cache_size
        )
    elif args.command == 'get':
        get_key_from_repo(
            key=args.key,
            ref=args.ref,
            prolly_dir=args.dir,
            cache_size=args.cache_size
        )
    elif args.command == 'set':
        set_key_in_repo(
            key=args.key,
            value=args.value,
            message=args.message,
            prolly_dir=args.dir,
            cache_size=args.cache_size,
            author=args.author
        )
    elif args.command == 'gc':
        gc_repo(
            prolly_dir=args.dir,
            cache_size=args.cache_size,
            dry_run=args.dry_run
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
