"""
Schema-aware database diffing for ProllyTree databases.

Provides high-level diff operations that understand table schemas and track
which columns changed across modified rows.
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from .db import DB, Table
from .tree import ProllyTree


@dataclass
class RowDiff:
    """Represents a diff for a single row."""
    primary_key: str
    status: str  # 'added', 'removed', 'modified'
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    changed_columns: Set[str] = field(default_factory=set)


@dataclass
class TableDiff:
    """Represents a diff for a single table."""
    table_name: str
    schema_changed: bool = False
    old_schema: Optional[Table] = None
    new_schema: Optional[Table] = None
    added_rows: List[RowDiff] = field(default_factory=list)
    removed_rows: List[RowDiff] = field(default_factory=list)
    modified_rows: List[RowDiff] = field(default_factory=list)
    changed_columns: Set[str] = field(default_factory=set)  # Union of all changed columns
    column_change_counts: Dict[str, int] = field(default_factory=dict)  # Column -> count of rows where it changed

    @property
    def has_changes(self) -> bool:
        """Check if this table has any changes."""
        return (self.schema_changed or
                len(self.added_rows) > 0 or
                len(self.removed_rows) > 0 or
                len(self.modified_rows) > 0)

    def summary(self) -> str:
        """Get a summary string of changes."""
        parts = []
        if self.schema_changed:
            parts.append("schema changed")
        if self.added_rows:
            parts.append(f"+{len(self.added_rows)} rows")
        if self.removed_rows:
            parts.append(f"-{len(self.removed_rows)} rows")
        if self.modified_rows:
            parts.append(f"~{len(self.modified_rows)} rows")
        if self.changed_columns:
            parts.append(f"columns: {', '.join(sorted(self.changed_columns))}")
        return ", ".join(parts) if parts else "no changes"

    def column_stats_summary(self) -> str:
        """Get a summary of column change statistics."""
        if not self.column_change_counts:
            return "no column changes"

        # Sort by count descending, then by name
        sorted_cols = sorted(
            self.column_change_counts.items(),
            key=lambda x: (-x[1], x[0])
        )

        lines = []
        for col, count in sorted_cols:
            pct = (count / len(self.modified_rows) * 100) if self.modified_rows else 0
            lines.append(f"{col}: {count:,} rows ({pct:.1f}%)")

        return "\n".join(lines)


@dataclass
class DBDiff:
    """Represents a diff between two databases."""
    added_tables: List[str] = field(default_factory=list)
    removed_tables: List[str] = field(default_factory=list)
    modified_tables: Dict[str, TableDiff] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (len(self.added_tables) > 0 or
                len(self.removed_tables) > 0 or
                any(td.has_changes for td in self.modified_tables.values()))

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all changes."""
        return {
            'added_tables': self.added_tables,
            'removed_tables': self.removed_tables,
            'modified_tables': {
                name: diff.summary()
                for name, diff in self.modified_tables.items()
                if diff.has_changes
            }
        }


def diff_table(old_db: DB, new_db: DB, table_name: str) -> TableDiff:
    """
    Diff a single table between two databases.

    Args:
        old_db: The old database state
        new_db: The new database state
        table_name: Name of the table to diff

    Returns:
        TableDiff object with detailed changes
    """
    diff = TableDiff(table_name=table_name)

    # Get schemas
    old_schema = old_db.get_table(table_name)
    new_schema = new_db.get_table(table_name)

    diff.old_schema = old_schema
    diff.new_schema = new_schema

    # Check for schema changes
    if old_schema and new_schema:
        diff.schema_changed = (
            old_schema.columns != new_schema.columns or
            old_schema.types != new_schema.types or
            old_schema.primary_key != new_schema.primary_key
        )

    # If schemas differ significantly, we can't do a detailed row diff
    if diff.schema_changed and old_schema and new_schema:
        if old_schema.columns != new_schema.columns:
            # Column set changed - can't reliably diff rows
            return diff

    # Get working schema (prefer new if available)
    schema = new_schema or old_schema
    if not schema:
        return diff

    # Load all rows from both databases
    old_rows: Dict[str, Dict[str, Any]] = {}
    new_rows: Dict[str, Dict[str, Any]] = {}

    # Load old rows
    if old_schema:
        for key, row_dict in old_db.read_rows(table_name, reconstruct=True):
            # Extract PK from key: /d/table_name/pk_value
            key_str = key.decode('utf-8')
            pk_value = key_str.split('/', 3)[3]  # Get everything after /d/table_name/
            old_rows[pk_value] = row_dict

    # Load new rows
    if new_schema:
        for key, row_dict in new_db.read_rows(table_name, reconstruct=True):
            key_str = key.decode('utf-8')
            pk_value = key_str.split('/', 3)[3]
            new_rows[pk_value] = row_dict

    # Find added, removed, and modified rows
    old_pks = set(old_rows.keys())
    new_pks = set(new_rows.keys())

    # Added rows
    for pk in new_pks - old_pks:
        diff.added_rows.append(RowDiff(
            primary_key=pk,
            status='added',
            new_values=new_rows[pk]
        ))

    # Removed rows
    for pk in old_pks - new_pks:
        diff.removed_rows.append(RowDiff(
            primary_key=pk,
            status='removed',
            old_values=old_rows[pk]
        ))

    # Modified rows
    for pk in old_pks & new_pks:
        old_row = old_rows[pk]
        new_row = new_rows[pk]

        # Find changed columns
        changed_cols = set()
        for col in schema.columns:
            old_val = old_row.get(col)
            new_val = new_row.get(col)
            if old_val != new_val:
                changed_cols.add(col)

        if changed_cols:
            diff.modified_rows.append(RowDiff(
                primary_key=pk,
                status='modified',
                old_values=old_row,
                new_values=new_row,
                changed_columns=changed_cols
            ))
            diff.changed_columns.update(changed_cols)

            # Update column change counts
            for col in changed_cols:
                diff.column_change_counts[col] = diff.column_change_counts.get(col, 0) + 1

    return diff


def diff_db(old_db: DB, new_db: DB, tables: Optional[List[str]] = None) -> DBDiff:
    """
    Diff two databases.

    Args:
        old_db: The old database state
        new_db: The new database state
        tables: Optional list of table names to diff. If None, diff all tables.

    Returns:
        DBDiff object with detailed changes across all tables
    """
    diff = DBDiff()

    # Get table lists
    old_tables = set(old_db.list_tables())
    new_tables = set(new_db.list_tables())

    # Filter if specific tables requested
    if tables:
        tables_set = set(tables)
        old_tables &= tables_set
        new_tables &= tables_set

    # Find added and removed tables
    diff.added_tables = sorted(new_tables - old_tables)
    diff.removed_tables = sorted(old_tables - new_tables)

    # Diff common tables
    common_tables = old_tables & new_tables
    for table_name in sorted(common_tables):
        table_diff = diff_table(old_db, new_db, table_name)
        if table_diff.has_changes:
            diff.modified_tables[table_name] = table_diff

    return diff


def diff_commits(store, old_commit_hash: bytes, new_commit_hash: bytes,
                 pattern: float = 0.01, seed: int = 42,
                 tables: Optional[List[str]] = None) -> DBDiff:
    """
    Diff two commits (assuming each commit points to a DB).

    Args:
        store: BlockStore instance to use
        old_commit_hash: Hash of the old commit
        new_commit_hash: Hash of the new commit
        pattern: ProllyTree pattern (should match the pattern used in commits)
        seed: ProllyTree seed (should match the seed used in commits)
        tables: Optional list of table names to diff

    Returns:
        DBDiff object with detailed changes
    """
    # Load old DB from old commit's tree root
    # Note: This assumes we can get the tree root from the commit
    # In practice, you'd need to load the commit and get its tree_root
    # For now, we assume old_commit_hash and new_commit_hash ARE tree roots

    old_db = DB(store=store, pattern=pattern, seed=seed)
    old_tree = ProllyTree(pattern=pattern, seed=seed, store=store)
    old_root = store.get_node(old_commit_hash)
    if old_root:
        old_tree.root = old_root
        old_db.tree = old_tree

    new_db = DB(store=store, pattern=pattern, seed=seed)
    new_tree = ProllyTree(pattern=pattern, seed=seed, store=store)
    new_root = store.get_node(new_commit_hash)
    if new_root:
        new_tree.root = new_root
        new_db.tree = new_tree

    return diff_db(old_db, new_db, tables=tables)
