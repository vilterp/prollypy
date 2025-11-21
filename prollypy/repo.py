"""
Repository abstraction for ProllyPy.

Provides git-like version control capabilities on top of ProllyTree.
Includes commit tracking, branching, and reference management.
"""

import time
import heapq
from typing import Optional, List, Dict, Tuple, Iterator, Set

from .store import BlockStore, RemoteStore
from .commit_graph_store import Commit, CommitGraphStore
from .commonality import collect_node_hashes


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
            pattern = head_commit.pattern if head_commit else 0.01
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
        - "HEAD~" or "HEAD~N" - resolves to the Nth parent of HEAD (default N=1)
        - Full commit hashes (64 hex characters)
        - Partial commit hashes (e.g., "341e719a") - must match exactly one commit

        Args:
            ref: Branch name, "HEAD", or commit hash (full or partial hex string)

        Returns:
            Commit hash bytes, or None if not found
        """
        # Handle HEAD~ syntax (HEAD~, HEAD~1, HEAD~2, etc.)
        if ref.startswith("HEAD~"):
            # Parse the number of parents to go back
            if ref == "HEAD~":
                n = 1
            else:
                try:
                    n = int(ref[5:])  # Everything after "HEAD~"
                except ValueError:
                    return None

            # Start with HEAD
            current_commit_hash = None
            head_commit, _ = self.get_head()
            if head_commit is None:
                return None
            current_commit_hash = head_commit.compute_hash()

            # Walk back n parents
            for _ in range(n):
                current_commit = self.get_commit(current_commit_hash)
                if current_commit is None or not current_commit.parents:
                    return None  # No parent available
                # Take the first parent
                current_commit_hash = current_commit.parents[0]

            return current_commit_hash

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

    def get_nodes_to_push(self, base_commit: Optional[bytes] = None) -> Set[bytes]:
        """
        Get all nodes that need to be pushed to a remote.

        Returns nodes reachable from HEAD but not from base_commit.
        If base_commit is None, returns all nodes reachable from HEAD.

        Args:
            base_commit: Hash of the last commit already in remote (optional)

        Returns:
            Set of node hashes to push
        """
        # Get HEAD commit
        head_commit, _ = self.get_head()
        if head_commit is None:
            return set()

        # Collect all tree roots from HEAD to base_commit
        new_tree_roots: Set[bytes] = set()
        for commit_hash, commit in self.log():
            new_tree_roots.add(commit.tree_root)
            # If we hit base_commit, stop (don't include it)
            if base_commit and commit_hash == base_commit:
                new_tree_roots.discard(commit.tree_root)
                break

        # Collect all nodes reachable from new tree roots
        new_nodes: Set[bytes] = set()
        for tree_root in new_tree_roots:
            nodes = collect_node_hashes(self.block_store, tree_root)
            new_nodes.update(nodes)

        # If base_commit specified, subtract nodes reachable from it
        if base_commit:
            base_commit_obj = self.get_commit(base_commit)
            if base_commit_obj is None:
                raise ValueError(f"Base commit not found: {base_commit.hex()}")

            # Collect all tree roots reachable from base_commit
            base_tree_roots: Set[bytes] = set()
            for commit_hash, commit in self.log(start_ref=base_commit.hex()):
                base_tree_roots.add(commit.tree_root)

            # Collect all nodes reachable from base
            base_nodes: Set[bytes] = set()
            for tree_root in base_tree_roots:
                nodes = collect_node_hashes(self.block_store, tree_root)
                base_nodes.update(nodes)

            # Subtract to get nodes to push
            return new_nodes - base_nodes

        return new_nodes

    def push(
        self,
        remote: RemoteStore,
        base_commit: Optional[bytes] = None
    ) -> Tuple[int, Iterator[bytes]]:
        """
        Push nodes to a remote store.

        Returns a tuple of (total_count, iterator) where the iterator yields
        each node hash as it's successfully pushed. This allows the caller
        to render a progress bar.

        Args:
            remote: Remote store to push nodes to
            base_commit: Hash of last commit already in remote (optional)

        Returns:
            Tuple of (total_nodes_to_push, iterator_of_pushed_hashes)
        """
        # Get nodes to push
        nodes_to_push = self.get_nodes_to_push(base_commit)
        total = len(nodes_to_push)

        def push_generator() -> Iterator[bytes]:
            for node_hash in nodes_to_push:
                node = self.block_store.get_node(node_hash)
                if node is None:
                    continue

                if remote.put_node(node_hash, node):
                    yield node_hash

        return (total, push_generator())
