"""
Commit graph store implementations for ProllyPy.

Provides storage backends for commits and references in the version control system.
"""

import time
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict


@dataclass
class Commit:
    """
    Represents a commit in the version control system.

    Each commit points to a prolly tree root and zero or more parent commits.
    """
    tree_root: bytes  # Hash of the prolly tree root node
    parents: List[bytes]  # List of parent commit hashes
    message: str  # Commit message
    timestamp: float  # Unix timestamp
    author: str  # Author name/email
    pattern: float = 0.01  # Split probability for the prolly tree
    seed: int = 42  # Seed for rolling hash function

    def to_dict(self) -> Dict:
        """Convert commit to dictionary for serialization."""
        return {
            'tree_root': self.tree_root.hex(),
            'parents': [p.hex() for p in self.parents],
            'message': self.message,
            'timestamp': self.timestamp,
            'author': self.author,
            'pattern': self.pattern,
            'seed': self.seed
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Commit':
        """Create commit from dictionary."""
        return cls(
            tree_root=bytes.fromhex(data['tree_root']),
            parents=[bytes.fromhex(p) for p in data['parents']],
            message=data['message'],
            timestamp=data['timestamp'],
            author=data['author'],
            pattern=data.get('pattern', 0.01),  # Default for backward compatibility
            seed=data.get('seed', 42)  # Default for backward compatibility
        )

    def compute_hash(self) -> bytes:
        """Compute the hash of this commit based on its contents."""
        # Create a deterministic serialization
        serialized = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).digest()


class CheckoutStore(Protocol):
    """Protocol for HEAD management (checkout operations)."""

    def set_head(self, ref_name: str) -> None:
        """Set HEAD to point to a branch name."""
        ...

    def get_head(self) -> Optional[str]:
        """Get the branch name that HEAD points to. Returns None if not set."""
        ...


class CommitGraphStore(Protocol):
    """Protocol for storing commits and references."""

    def put_commit(self, commit_hash: bytes, commit: Commit) -> None:
        """Store a commit by its hash."""
        ...

    def get_commit(self, commit_hash: bytes) -> Optional[Commit]:
        """Retrieve a commit by its hash. Returns None if not found."""
        ...

    def get_parents(self, commit_hash: bytes) -> List[bytes]:
        """Get parent commit hashes for a given commit."""
        ...

    def set_ref(self, name: str, commit_hash: bytes) -> None:
        """Set a reference (branch/tag) to point to a commit."""
        ...

    def get_ref(self, name: str) -> Optional[bytes]:
        """Get the commit hash for a reference. Returns None if not found."""
        ...

    def list_refs(self) -> Dict[str, bytes]:
        """List all references and their commit hashes."""
        ...

    def find_commit_by_prefix(self, prefix: str) -> Optional[bytes]:
        """
        Find a commit by its hash prefix (partial hash).

        Args:
            prefix: Hex string prefix of the commit hash

        Returns:
            Full commit hash if exactly one match is found, None otherwise
        """
        ...


class LocalCommitGraphStore(CommitGraphStore, CheckoutStore, Protocol):
    """Protocol for local commit storage that includes both CommitGraphStore and CheckoutStore."""
    pass


class MemoryCommitGraphStore(LocalCommitGraphStore):
    """In-memory implementation of LocalCommitGraphStore."""

    def __init__(self):
        self.commits: Dict[bytes, Commit] = {}
        self.refs: Dict[str, bytes] = {}
        self.head: Optional[str] = None

    def put_commit(self, commit_hash: bytes, commit: Commit) -> None:
        """Store a commit in memory."""
        self.commits[commit_hash] = commit

    def get_commit(self, commit_hash: bytes) -> Optional[Commit]:
        """Retrieve a commit from memory."""
        return self.commits.get(commit_hash)

    def get_parents(self, commit_hash: bytes) -> List[bytes]:
        """Get parent commit hashes."""
        commit = self.get_commit(commit_hash)
        return commit.parents if commit else []

    def set_ref(self, name: str, commit_hash: bytes) -> None:
        """Set a reference in memory."""
        self.refs[name] = commit_hash

    def get_ref(self, name: str) -> Optional[bytes]:
        """Get a reference from memory."""
        return self.refs.get(name)

    def list_refs(self) -> Dict[str, bytes]:
        """List all references."""
        return dict(self.refs)

    def set_head(self, ref_name: str) -> None:
        """Set HEAD to point to a branch name."""
        self.head = ref_name

    def get_head(self) -> Optional[str]:
        """Get the branch name that HEAD points to."""
        return self.head

    def find_commit_by_prefix(self, prefix: str) -> Optional[bytes]:
        """Find a commit by its hash prefix."""
        prefix_lower = prefix.lower()
        matches = [
            commit_hash
            for commit_hash in self.commits.keys()
            if commit_hash.hex().startswith(prefix_lower)
        ]
        return matches[0] if len(matches) == 1 else None


class SqliteCommitGraphStore(LocalCommitGraphStore):
    """SQLite-based implementation of LocalCommitGraphStore."""

    def __init__(self, db_path: str):
        """
        Initialize SQLite commit graph store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # Commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                hash BLOB PRIMARY KEY,
                tree_root BLOB NOT NULL,
                message TEXT NOT NULL,
                timestamp REAL NOT NULL,
                author TEXT NOT NULL,
                pattern REAL NOT NULL DEFAULT 0.01,
                seed INTEGER NOT NULL DEFAULT 42
            )
        """)

        # Commit parents table (many-to-many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commit_parents (
                commit_hash BLOB NOT NULL,
                parent_hash BLOB NOT NULL,
                parent_index INTEGER NOT NULL,
                PRIMARY KEY (commit_hash, parent_index),
                FOREIGN KEY (commit_hash) REFERENCES commits(hash)
            )
        """)

        # Refs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refs (
                name TEXT PRIMARY KEY,
                commit_hash BLOB NOT NULL,
                FOREIGN KEY (commit_hash) REFERENCES commits(hash)
            )
        """)

        # Metadata table for HEAD and other repo settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        self.conn.commit()

    def put_commit(self, commit_hash: bytes, commit: Commit) -> None:
        """Store a commit in SQLite."""
        cursor = self.conn.cursor()

        # Insert commit
        cursor.execute("""
            INSERT OR REPLACE INTO commits (hash, tree_root, message, timestamp, author, pattern, seed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (commit_hash, commit.tree_root, commit.message, commit.timestamp, commit.author, commit.pattern, commit.seed))

        # Delete existing parents (in case of replacement)
        cursor.execute("DELETE FROM commit_parents WHERE commit_hash = ?", (commit_hash,))

        # Insert parents
        for idx, parent_hash in enumerate(commit.parents):
            cursor.execute("""
                INSERT INTO commit_parents (commit_hash, parent_hash, parent_index)
                VALUES (?, ?, ?)
            """, (commit_hash, parent_hash, idx))

        self.conn.commit()

    def get_commit(self, commit_hash: bytes) -> Optional[Commit]:
        """Retrieve a commit from SQLite."""
        cursor = self.conn.cursor()

        # Get commit data
        cursor.execute("""
            SELECT tree_root, message, timestamp, author, pattern, seed
            FROM commits
            WHERE hash = ?
        """, (commit_hash,))

        row = cursor.fetchone()
        if not row:
            return None

        tree_root, message, timestamp, author, pattern, seed = row

        # Get parents
        cursor.execute("""
            SELECT parent_hash
            FROM commit_parents
            WHERE commit_hash = ?
            ORDER BY parent_index
        """, (commit_hash,))

        parents = [row[0] for row in cursor.fetchall()]

        return Commit(
            tree_root=tree_root,
            parents=parents,
            message=message,
            timestamp=timestamp,
            author=author,
            pattern=pattern,
            seed=seed
        )

    def get_parents(self, commit_hash: bytes) -> List[bytes]:
        """Get parent commit hashes from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT parent_hash
            FROM commit_parents
            WHERE commit_hash = ?
            ORDER BY parent_index
        """, (commit_hash,))

        return [row[0] for row in cursor.fetchall()]

    def set_ref(self, name: str, commit_hash: bytes) -> None:
        """Set a reference in SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO refs (name, commit_hash)
            VALUES (?, ?)
        """, (name, commit_hash))
        self.conn.commit()

    def get_ref(self, name: str) -> Optional[bytes]:
        """Get a reference from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT commit_hash FROM refs WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def list_refs(self) -> Dict[str, bytes]:
        """List all references from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, commit_hash FROM refs")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def set_head(self, ref_name: str) -> None:
        """Set HEAD to point to a branch name."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('HEAD', ?)
        """, (ref_name,))
        self.conn.commit()

    def get_head(self) -> Optional[str]:
        """Get the branch name that HEAD points to."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'HEAD'")
        row = cursor.fetchone()
        return row[0] if row else None

    def find_commit_by_prefix(self, prefix: str) -> Optional[bytes]:
        """Find a commit by its hash prefix."""
        cursor = self.conn.cursor()
        # Use LIKE with hex prefix pattern
        # Convert prefix to uppercase to match SQLite's hex() output
        prefix_upper = prefix.upper()
        pattern = f"{prefix_upper}%"
        cursor.execute("""
            SELECT hash FROM commits
            WHERE hex(hash) LIKE ?
        """, (pattern,))
        matches = cursor.fetchall()
        return matches[0][0] if len(matches) == 1 else None

    def close(self):
        """Close the database connection."""
        self.conn.close()
