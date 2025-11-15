"""
Database abstraction layer for ProllyTree.

Provides a high-level database interface on top of the ProllyTree key-value store.
"""

import json
from typing import List, Dict, Any, Optional, Iterator, Tuple
from .tree import ProllyTree
from .store import Store, create_store_from_spec


class Table:
    """Represents a table schema."""

    def __init__(self, name: str, columns: List[str], types: List[str], primary_key: List[str]):
        self.name = name
        self.columns = columns
        self.types = types
        self.primary_key = primary_key

    def to_dict(self) -> Dict[str, Any]:
        """Convert table schema to dictionary."""
        return {
            'columns': self.columns,
            'types': self.types,
            'primary_key': self.primary_key
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'Table':
        """Create Table from dictionary."""
        return cls(
            name=name,
            columns=data['columns'],
            types=data['types'],
            primary_key=data['primary_key']
        )


class DB:
    """
    Database abstraction layer for ProllyTree.

    Stores:
    - Table schemas at /s/<table_name>
    - Table data at /d/<table_name>/<primary_key>
    """

    def __init__(self, store: Store, pattern: float = 0.0001, seed: int = 42, validate: bool = False):
        """
        Initialize database with a Store instance.

        Args:
            store: Storage backend instance
            pattern: ProllyTree split pattern
            seed: Random seed for rolling hash
            validate: If True, validate tree structure during operations (slower)
        """
        self.tree = ProllyTree(pattern=pattern, seed=seed, store=store, validate=validate)
        self._validate_after_batch = False  # Internal flag for validation control

    def create_table(self, name: str, columns: List[str], types: List[str],
                     primary_key: List[str]) -> Table:
        """
        Create a new table schema.

        Args:
            name: Table name
            columns: List of column names
            types: List of column types
            primary_key: List of primary key column names

        Returns:
            Created Table instance
        """
        table = Table(name, columns, types, primary_key)

        # Store schema
        schema_key = f"/s/{name}".encode('utf-8')
        schema_value = json.dumps(table.to_dict(), separators=(',', ':')).encode('utf-8')
        self.tree.insert_batch([(schema_key, schema_value)], verbose=False)

        return table

    def get_table(self, name: str) -> Optional[Table]:
        """
        Get table schema by name.

        Args:
            name: Table name

        Returns:
            Table instance or None if not found
        """
        schema_key = f"/s/{name}".encode('utf-8')
        for key, value in self.tree.items(schema_key):
            if key == schema_key:
                return Table.from_dict(name, json.loads(value.decode('utf-8')))
        return None

    def list_tables(self) -> List[str]:
        """
        List all table names.

        Returns:
            List of table names
        """
        tables = []
        for key, _ in self.tree.items(b'/s/'):
            table_name = key[3:].decode('utf-8')  # Remove '/s/' prefix and decode
            tables.append(table_name)
        return tables

    def insert_rows(self, table_name: str, rows: Iterator[Tuple],
                    batch_size: int = 1000, verbose: bool = False) -> int:
        """
        Insert rows into a table.

        Args:
            table_name: Name of the table
            rows: Iterator of row tuples (values in column order)
            batch_size: Number of rows per batch
            verbose: Whether to print verbose output

        Returns:
            Number of rows inserted
        """
        table = self.get_table(table_name)
        if not table:
            raise ValueError(f"Table {table_name} does not exist")

        total_inserted = 0
        batch = []

        for row in rows:
            # Build primary key from row values
            # Special case: if primary_key is ["rowid"], the first element is the rowid
            if table.primary_key == ["rowid"]:
                pk_parts = [str(row[0])]
            else:
                pk_indices = [table.columns.index(col) for col in table.primary_key]
                pk_parts = [str(row[i]) for i in pk_indices]
            pk_value = "/".join(pk_parts)

            # Create key-value pair
            key = f"/d/{table_name}/{pk_value}".encode('utf-8')
            value = json.dumps(list(row), separators=(',', ':')).encode('utf-8')
            batch.append((key, value))

            # Insert batch when full
            if len(batch) >= batch_size:
                batch.sort(key=lambda x: x[0])
                self.tree.insert_batch(batch, verbose=verbose)
                total_inserted += len(batch)
                batch = []

        # Insert remaining rows
        if batch:
            batch.sort(key=lambda x: x[0])
            self.tree.insert_batch(batch, verbose=verbose)
            total_inserted += len(batch)

        return total_inserted

    def read_rows(self, table_name: str, prefix: str = "",
                  reconstruct: bool = True) -> Iterator[Tuple[bytes, Any]]:
        """
        Read rows from a table.

        Args:
            table_name: Name of the table
            prefix: Additional prefix filter (appended to /d/table_name/)
            reconstruct: If True, return dicts; if False, return raw arrays

        Yields:
            Tuples of (key, row_data) where key is bytes and row_data is dict if reconstruct=True, else list
        """
        table = self.get_table(table_name) if reconstruct else None
        data_prefix = f"/d/{table_name}/{prefix}".encode('utf-8')

        for key, value in self.tree.items(data_prefix):
            row_values = json.loads(value.decode('utf-8'))

            if reconstruct and table:
                # Reconstruct as dictionary
                row_dict = dict(zip(table.columns, row_values))
                yield (key, row_dict)
            else:
                # Return raw array
                yield (key, row_values)

    def get_root_hash(self) -> bytes:
        """
        Get the current root hash of the tree.

        Returns:
            Root hash as bytes
        """
        return self.tree._hash_node(self.tree.root)

    def get_store(self) -> Store:
        """
        Get the underlying store.

        Returns:
            Store instance
        """
        return self.tree.store

    def count_rows(self, table_name: str) -> int:
        """
        Count rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows
        """
        count = 0
        for _ in self.read_rows(table_name, reconstruct=False):
            count += 1
        return count
