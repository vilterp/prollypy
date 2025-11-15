"""
Repository abstraction for ProllyPy.

Provides git-like version control capabilities on top of ProllyTree.
Includes commit tracking, branching, and reference management.
"""

import time
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict, Tuple
from .store import BlockStore


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

    def to_dict(self) -> Dict:
        """Convert commit to dictionary for serialization."""
        return {
            'tree_root': self.tree_root.hex(),
            'parents': [p.hex() for p in self.parents],
            'message': self.message,
            'timestamp': self.timestamp,
            'author': self.author
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Commit':
        """Create commit from dictionary."""
        return cls(
            tree_root=bytes.fromhex(data['tree_root']),
            parents=[bytes.fromhex(p) for p in data['parents']],
            message=data['message'],
            timestamp=data['timestamp'],
            author=data['author']
        )

    def compute_hash(self) -> bytes:
        """Compute the hash of this commit based on its contents."""
        # Create a deterministic serialization
        serialized = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).digest()


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


class MemoryCommitGraphStore:
    """In-memory implementation of CommitGraphStore."""

    def __init__(self):
        self.commits: Dict[bytes, Commit] = {}
        self.refs: Dict[str, bytes] = {}

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


class SqliteCommitGraphStore:
    """SQLite-based implementation of CommitGraphStore."""

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
                author TEXT NOT NULL
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

        self.conn.commit()

    def put_commit(self, commit_hash: bytes, commit: Commit) -> None:
        """Store a commit in SQLite."""
        cursor = self.conn.cursor()

        # Insert commit
        cursor.execute("""
            INSERT OR REPLACE INTO commits (hash, tree_root, message, timestamp, author)
            VALUES (?, ?, ?, ?, ?)
        """, (commit_hash, commit.tree_root, commit.message, commit.timestamp, commit.author))

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
            SELECT tree_root, message, timestamp, author
            FROM commits
            WHERE hash = ?
        """, (commit_hash,))

        row = cursor.fetchone()
        if not row:
            return None

        tree_root, message, timestamp, author = row

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
            author=author
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

    def close(self):
        """Close the database connection."""
        self.conn.close()


class Repo:
    """
    Repository for version-controlled ProllyTrees.

    Provides git-like operations: commit, branch, and reference management.
    """

    def __init__(self, block_store: BlockStore, commit_graph_store: CommitGraphStore,
                 default_author: str = "unknown"):
        """
        Initialize repository.

        Args:
            block_store: Storage for tree nodes
            commit_graph_store: Storage for commits and refs
            default_author: Default author for commits
        """
        self.block_store = block_store
        self.commit_graph_store = commit_graph_store
        self.default_author = default_author
        self._current_ref = "main"  # Default branch

    def get_head(self) -> Tuple[Optional[Commit], str]:
        """
        Get the current HEAD commit and ref name.

        Returns:
            Tuple of (commit, ref_name). Commit may be None if ref doesn't exist yet.
        """
        commit_hash = self.commit_graph_store.get_ref(self._current_ref)
        if commit_hash is None:
            return (None, self._current_ref)

        commit = self.commit_graph_store.get_commit(commit_hash)
        return (commit, self._current_ref)

    def create_branch(self, name: str, from_commit: Optional[bytes] = None) -> None:
        """
        Create a new branch.

        Args:
            name: Branch name
            from_commit: Commit hash to branch from. If None, uses current HEAD.

        Raises:
            ValueError: If branch already exists or from_commit is invalid
        """
        # Check if branch already exists
        if self.commit_graph_store.get_ref(name) is not None:
            raise ValueError(f"Branch '{name}' already exists")

        # Determine commit to branch from
        if from_commit is None:
            head_commit, _ = self.get_head()
            if head_commit is None:
                raise ValueError("Cannot create branch: no commits exist yet")
            from_commit = head_commit.compute_hash()

        # Verify the commit exists
        if self.commit_graph_store.get_commit(from_commit) is None:
            raise ValueError(f"Commit {from_commit.hex()} does not exist")

        # Create the branch
        self.commit_graph_store.set_ref(name, from_commit)

    def checkout(self, ref_name: str) -> None:
        """
        Switch to a different branch.

        Args:
            ref_name: Name of the branch to checkout

        Raises:
            ValueError: If ref doesn't exist
        """
        if self.commit_graph_store.get_ref(ref_name) is None:
            raise ValueError(f"Ref '{ref_name}' does not exist")

        self._current_ref = ref_name

    def commit(self, new_head_tree: bytes, message: str, author: Optional[str] = None) -> Commit:
        """
        Create a new commit with the given tree root.

        Args:
            new_head_tree: Hash of the prolly tree root for this commit
            message: Commit message
            author: Author name/email. Uses default if not provided.

        Returns:
            The newly created Commit object
        """
        # Get current HEAD as parent
        head_commit, ref_name = self.get_head()
        parents = [head_commit.compute_hash()] if head_commit else []

        # Create new commit
        new_commit = Commit(
            tree_root=new_head_tree,
            parents=parents,
            message=message,
            timestamp=time.time(),
            author=author or self.default_author
        )

        # Compute commit hash and store it
        commit_hash = new_commit.compute_hash()
        self.commit_graph_store.put_commit(commit_hash, new_commit)

        # Update current ref to point to new commit
        self.commit_graph_store.set_ref(ref_name, commit_hash)

        return new_commit

    def get_commit(self, commit_hash: bytes) -> Optional[Commit]:
        """Get a commit by its hash."""
        return self.commit_graph_store.get_commit(commit_hash)

    def list_branches(self) -> Dict[str, bytes]:
        """List all branches and their commit hashes."""
        return self.commit_graph_store.list_refs()
