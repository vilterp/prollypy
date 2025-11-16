"""
Repository abstraction for ProllyPy.

Provides git-like version control capabilities on top of ProllyTree.
Includes commit tracking, branching, and reference management.
"""

import time
import hashlib
import json
import sqlite3
import heapq
from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict, Tuple, Iterator, Set
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
    pattern: float = 0.0001  # Split probability for the prolly tree
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
            pattern=data.get('pattern', 0.0001),  # Default for backward compatibility
            seed=data.get('seed', 42)  # Default for backward compatibility
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

    def set_head(self, ref_name: str) -> None:
        """Set HEAD to point to a branch name."""
        ...

    def get_head(self) -> Optional[str]:
        """Get the branch name that HEAD points to. Returns None if not set."""
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


class MemoryCommitGraphStore:
    """In-memory implementation of CommitGraphStore."""

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
                author TEXT NOT NULL,
                pattern REAL NOT NULL DEFAULT 0.0001,
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

    @classmethod
    def init_empty(cls, block_store: BlockStore, commit_graph_store: CommitGraphStore,
                   default_author: str = "unknown") -> 'Repo':
        """
        Initialize a new empty repository with an initial commit.

        Args:
            block_store: Storage for tree nodes
            commit_graph_store: Storage for commits and refs
            default_author: Default author for commits

        Returns:
            A new Repo instance
        """
        from .tree import ProllyTree
        from .node import Node

        repo = cls(block_store, commit_graph_store, default_author)

        # Create empty tree
        empty_tree = ProllyTree(store=block_store)
        tree_root = empty_tree._hash_node(empty_tree.root)

        # Store the empty root node in the block store
        block_store.put_node(tree_root, empty_tree.root)

        # Create initial commit
        initial_commit = Commit(
            tree_root=tree_root,
            parents=[],
            message="Initial commit",
            timestamp=time.time(),
            author=default_author
        )

        commit_hash = initial_commit.compute_hash()
        commit_graph_store.put_commit(commit_hash, initial_commit)

        # Create main branch and set HEAD
        commit_graph_store.set_ref("main", commit_hash)
        commit_graph_store.set_head("main")

        return repo

    def get_head(self) -> Tuple[Optional[Commit], str]:
        """
        Get the current HEAD commit and ref name.

        Returns:
            Tuple of (commit, ref_name). Commit may be None if ref doesn't exist yet.
        """
        current_ref = self.commit_graph_store.get_head()
        if current_ref is None:
            # HEAD not set, default to "main" and set it
            current_ref = "main"
            self.commit_graph_store.set_head(current_ref)

        commit_hash = self.commit_graph_store.get_ref(current_ref)
        if commit_hash is None:
            return (None, current_ref)

        commit = self.commit_graph_store.get_commit(commit_hash)
        return (commit, current_ref)

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

        self.commit_graph_store.set_head(ref_name)

    def commit(self, new_head_tree: bytes, message: str, author: Optional[str] = None,
               pattern: Optional[float] = None, seed: Optional[int] = None) -> Commit:
        """
        Create a new commit with the given tree root.

        Args:
            new_head_tree: Hash of the prolly tree root for this commit
            message: Commit message
            author: Author name/email. Uses default if not provided.
            pattern: Split probability for the prolly tree. If None, inherits from parent.
            seed: Seed for rolling hash function. If None, inherits from parent.

        Returns:
            The newly created Commit object
        """
        # Get current HEAD as parent
        head_commit, ref_name = self.get_head()
        parents = [head_commit.compute_hash()] if head_commit else []

        # Inherit pattern and seed from parent if not specified
        if pattern is None:
            pattern = head_commit.pattern if head_commit else 0.0001
        if seed is None:
            seed = head_commit.seed if head_commit else 42

        # Create new commit
        new_commit = Commit(
            tree_root=new_head_tree,
            parents=parents,
            message=message,
            timestamp=time.time(),
            author=author or self.default_author,
            pattern=pattern,
            seed=seed
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

    def resolve_ref(self, ref: str) -> Optional[bytes]:
        """
        Resolve a ref name or commit hash to a commit hash.

        Supports:
        - Branch/tag names (e.g., "main", "develop")
        - "HEAD" - resolves to the commit of the current HEAD ref
        - Full commit hashes (64 hex characters)
        - Partial commit hashes (e.g., "341e719a") - must match exactly one commit

        Args:
            ref: Branch name, "HEAD", or commit hash (full or partial hex string)

        Returns:
            Commit hash bytes, or None if not found
        """
        # Handle HEAD specially
        if ref == "HEAD":
            head_commit, _ = self.get_head()
            if head_commit is None:
                return None
            return head_commit.compute_hash()

        # Try as a ref (branch/tag) first
        commit_hash = self.commit_graph_store.get_ref(ref)
        if commit_hash is not None:
            return commit_hash

        # Try as a full commit hash
        try:
            commit_hash = bytes.fromhex(ref)
            # Verify it exists
            if self.commit_graph_store.get_commit(commit_hash) is not None:
                return commit_hash
        except ValueError:
            pass

        # Try as a partial commit hash
        if all(c in '0123456789abcdefABCDEF' for c in ref):
            partial_match = self.commit_graph_store.find_commit_by_prefix(ref)
            if partial_match is not None:
                return partial_match

        return None

    def walk(self, frontier: Set[bytes]) -> Iterator[Tuple[bytes, Commit]]:
        """
        Walk the commit graph starting from a frontier set, yielding commits
        in reverse chronological order (most recent first).

        This method uses a priority queue to ensure commits are yielded in
        timestamp order (newest first), which is important for garbage collection
        and other operations that need to traverse the commit graph in a specific order.

        Args:
            frontier: Set of commit hashes to start walking from

        Yields:
            Tuples of (commit_hash, commit) in reverse chronological order
        """
        # Priority queue: (negative_timestamp, commit_hash)
        # We use negative timestamp because heapq is a min-heap
        # and we want the most recent (highest timestamp) first
        heap: List[Tuple[float, bytes]] = []
        visited: Set[bytes] = set()

        # Initialize heap with frontier commits
        for commit_hash in frontier:
            commit = self.commit_graph_store.get_commit(commit_hash)
            if commit is not None:
                # Use negative timestamp for max-heap behavior
                heapq.heappush(heap, (-commit.timestamp, commit_hash))

        # Process commits in timestamp order
        while heap:
            neg_timestamp, commit_hash = heapq.heappop(heap)

            # Skip if already visited
            if commit_hash in visited:
                continue

            visited.add(commit_hash)

            # Get the commit
            commit = self.commit_graph_store.get_commit(commit_hash)
            if commit is None:
                continue

            yield (commit_hash, commit)

            # Add parents to the heap
            for parent_hash in commit.parents:
                if parent_hash not in visited:
                    parent_commit = self.commit_graph_store.get_commit(parent_hash)
                    if parent_commit is not None:
                        heapq.heappush(heap, (-parent_commit.timestamp, parent_hash))

    def log(self, start_ref: Optional[str] = None, max_count: Optional[int] = None) -> Iterator[Tuple[bytes, Commit]]:
        """
        Walk the commit graph backwards from a starting point, yielding commits.

        This is implemented in terms of walk() by starting with a frontier of {HEAD}
        or the specified ref.

        Args:
            start_ref: Ref or commit hash to start from. If None, uses HEAD.
            max_count: Maximum number of commits to return. If None, returns all.

        Yields:
            Tuples of (commit_hash, commit) in reverse chronological order
        """
        # Determine starting commit
        if start_ref is None:
            head_commit, _ = self.get_head()
            if head_commit is None:
                return
            current_hash = head_commit.compute_hash()
        else:
            current_hash = self.resolve_ref(start_ref)
            if current_hash is None:
                return

        # Create frontier with just the starting commit
        frontier = {current_hash}

        # Walk and yield commits, respecting max_count
        count = 0
        for commit_hash, commit in self.walk(frontier):
            if max_count is not None and count >= max_count:
                break
            yield (commit_hash, commit)
            count += 1

    def get_reachable_tree_roots(self) -> Set[bytes]:
        """
        Get all tree roots that are reachable from any ref.

        This walks the commit graph starting from all refs and collects
        all tree roots. This is useful for garbage collection to identify
        which prolly tree nodes should be kept.

        Returns:
            Set of tree root hashes (as bytes) that are reachable
        """
        # Get all refs as the frontier
        refs = self.commit_graph_store.list_refs()
        if not refs:
            return set()

        frontier = set(refs.values())

        # Walk through all commits and collect tree roots
        tree_roots: Set[bytes] = set()
        for commit_hash, commit in self.walk(frontier):
            tree_roots.add(commit.tree_root)

        return tree_roots
