"""
Tests for Repo class and commit graph functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path

from prollypy.repo import Repo
from prollypy.commit_graph_store import (
    Commit,
    CommitGraphStore,
    MemoryCommitGraphStore,
    SqliteCommitGraphStore,
)
from prollypy.store import MemoryBlockStore
from prollypy.tree import ProllyTree


@pytest.fixture
def block_store():
    """Create a memory block store."""
    return MemoryBlockStore()


@pytest.fixture
def memory_commit_store():
    """Create a memory commit graph store."""
    return MemoryCommitGraphStore()


@pytest.fixture
def sqlite_commit_store():
    """Create a SQLite commit graph store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name

    store = SqliteCommitGraphStore(db_path)
    yield store
    store.close()
    os.unlink(db_path)


@pytest.fixture
def repo_memory(block_store, memory_commit_store):
    """Create a repo with memory stores."""
    return Repo(block_store, memory_commit_store, default_author="test@example.com")


@pytest.fixture
def repo_sqlite(block_store, sqlite_commit_store):
    """Create a repo with SQLite commit store."""
    return Repo(block_store, sqlite_commit_store, default_author="test@example.com")


class TestCommit:
    """Tests for Commit dataclass."""

    def test_commit_creation(self):
        """Test creating a commit."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[b'parent1', b'parent2'],
            message="Test commit",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        assert commit.tree_root == b'abc123'
        assert commit.parents == [b'parent1', b'parent2']
        assert commit.message == "Test commit"
        assert commit.timestamp == 1234567890.0
        assert commit.author == "test@example.com"

    def test_commit_to_dict(self):
        """Test converting commit to dictionary."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[b'parent1'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        d = commit.to_dict()
        assert d['tree_root'] == 'abc123'.encode().hex()
        assert d['parents'] == ['parent1'.encode().hex()]
        assert d['message'] == "Test"
        assert d['timestamp'] == 1234567890.0
        assert d['author'] == "test@example.com"

    def test_commit_from_dict(self):
        """Test creating commit from dictionary."""
        d = {
            'tree_root': 'abc123'.encode().hex(),
            'parents': ['parent1'.encode().hex()],
            'message': "Test",
            'timestamp': 1234567890.0,
            'author': "test@example.com"
        }

        commit = Commit.from_dict(d)
        assert commit.tree_root == b'abc123'
        assert commit.parents == [b'parent1']
        assert commit.message == "Test"
        assert commit.timestamp == 1234567890.0
        assert commit.author == "test@example.com"

    def test_commit_hash(self):
        """Test that commit hash is deterministic."""
        commit1 = Commit(
            tree_root=b'abc123',
            parents=[b'parent1'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        commit2 = Commit(
            tree_root=b'abc123',
            parents=[b'parent1'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        # Same content should produce same hash
        assert commit1.compute_hash() == commit2.compute_hash()

        # Different content should produce different hash
        commit3 = Commit(
            tree_root=b'different',
            parents=[b'parent1'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        assert commit1.compute_hash() != commit3.compute_hash()


class TestMemoryCommitGraphStore:
    """Tests for MemoryCommitGraphStore."""

    def test_put_get_commit(self, memory_commit_store):
        """Test storing and retrieving commits."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[],
            message="Initial commit",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        commit_hash = commit.compute_hash()
        memory_commit_store.put_commit(commit_hash, commit)

        retrieved = memory_commit_store.get_commit(commit_hash)
        assert retrieved is not None
        assert retrieved.tree_root == b'abc123'
        assert retrieved.message == "Initial commit"

    def test_get_nonexistent_commit(self, memory_commit_store):
        """Test getting a commit that doesn't exist."""
        assert memory_commit_store.get_commit(b'nonexistent') is None

    def test_get_parents(self, memory_commit_store):
        """Test getting parent commits."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[b'parent1', b'parent2'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        commit_hash = commit.compute_hash()
        memory_commit_store.put_commit(commit_hash, commit)

        parents = memory_commit_store.get_parents(commit_hash)
        assert parents == [b'parent1', b'parent2']

    def test_set_get_ref(self, memory_commit_store):
        """Test setting and getting references."""
        memory_commit_store.set_ref("main", b'commit123')
        assert memory_commit_store.get_ref("main") == b'commit123'

    def test_get_nonexistent_ref(self, memory_commit_store):
        """Test getting a ref that doesn't exist."""
        assert memory_commit_store.get_ref("nonexistent") is None

    def test_list_refs(self, memory_commit_store):
        """Test listing all refs."""
        memory_commit_store.set_ref("main", b'commit1')
        memory_commit_store.set_ref("develop", b'commit2')

        refs = memory_commit_store.list_refs()
        assert refs == {"main": b'commit1', "develop": b'commit2'}


class TestSqliteCommitGraphStore:
    """Tests for SqliteCommitGraphStore."""

    def test_put_get_commit(self, sqlite_commit_store):
        """Test storing and retrieving commits."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[],
            message="Initial commit",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        commit_hash = commit.compute_hash()
        sqlite_commit_store.put_commit(commit_hash, commit)

        retrieved = sqlite_commit_store.get_commit(commit_hash)
        assert retrieved is not None
        assert retrieved.tree_root == b'abc123'
        assert retrieved.message == "Initial commit"

    def test_get_nonexistent_commit(self, sqlite_commit_store):
        """Test getting a commit that doesn't exist."""
        assert sqlite_commit_store.get_commit(b'nonexistent') is None

    def test_commit_with_parents(self, sqlite_commit_store):
        """Test storing and retrieving commits with parents."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[b'parent1', b'parent2'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        commit_hash = commit.compute_hash()
        sqlite_commit_store.put_commit(commit_hash, commit)

        retrieved = sqlite_commit_store.get_commit(commit_hash)
        assert retrieved.parents == [b'parent1', b'parent2']

    def test_get_parents(self, sqlite_commit_store):
        """Test getting parent commits."""
        commit = Commit(
            tree_root=b'abc123',
            parents=[b'parent1', b'parent2'],
            message="Test",
            timestamp=1234567890.0,
            author="test@example.com"
        )

        commit_hash = commit.compute_hash()
        sqlite_commit_store.put_commit(commit_hash, commit)

        parents = sqlite_commit_store.get_parents(commit_hash)
        assert parents == [b'parent1', b'parent2']

    def test_set_get_ref(self, sqlite_commit_store):
        """Test setting and getting references."""
        sqlite_commit_store.set_ref("main", b'commit123')
        assert sqlite_commit_store.get_ref("main") == b'commit123'

    def test_get_nonexistent_ref(self, sqlite_commit_store):
        """Test getting a ref that doesn't exist."""
        assert sqlite_commit_store.get_ref("nonexistent") is None

    def test_list_refs(self, sqlite_commit_store):
        """Test listing all refs."""
        sqlite_commit_store.set_ref("main", b'commit1')
        sqlite_commit_store.set_ref("develop", b'commit2')

        refs = sqlite_commit_store.list_refs()
        assert refs == {"main": b'commit1', "develop": b'commit2'}

    def test_update_ref(self, sqlite_commit_store):
        """Test updating an existing ref."""
        sqlite_commit_store.set_ref("main", b'commit1')
        sqlite_commit_store.set_ref("main", b'commit2')

        assert sqlite_commit_store.get_ref("main") == b'commit2'


class TestRepo:
    """Tests for Repo class."""

    def test_initial_head(self, repo_memory):
        """Test that initial HEAD is None."""
        commit, ref = repo_memory.get_head()
        assert commit is None
        assert ref == "main"

    def test_first_commit(self, repo_memory, block_store):
        """Test creating the first commit."""
        # Create a simple tree
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root = tree._hash_node(tree.root)

        # Create first commit
        commit = repo_memory.commit(tree_root, "Initial commit")

        assert commit.tree_root == tree_root
        assert commit.parents == []
        assert commit.message == "Initial commit"
        assert commit.author == "test@example.com"

        # Verify HEAD points to new commit
        head_commit, ref = repo_memory.get_head()
        assert head_commit is not None
        assert head_commit.tree_root == tree_root
        assert ref == "main"

    def test_second_commit(self, repo_memory, block_store):
        """Test creating a second commit with parent."""
        # Create first tree and commit
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root1 = tree._hash_node(tree.root)
        commit1 = repo_memory.commit(tree_root1, "First commit")

        # Create second tree and commit
        tree.insert_batch([(b'key2', b'value2')])
        tree_root2 = tree._hash_node(tree.root)
        commit2 = repo_memory.commit(tree_root2, "Second commit")

        # Verify second commit has first commit as parent
        assert len(commit2.parents) == 1
        assert commit2.parents[0] == commit1.compute_hash()

        # Verify HEAD points to second commit
        head_commit, _ = repo_memory.get_head()
        assert head_commit.compute_hash() == commit2.compute_hash()

    def test_create_branch(self, repo_memory, block_store):
        """Test creating a new branch."""
        # Create initial commit
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root = tree._hash_node(tree.root)
        commit1 = repo_memory.commit(tree_root, "Initial commit")

        # Create a branch
        repo_memory.create_branch("develop")

        # Verify branch exists and points to same commit
        branch_hash = repo_memory.commit_graph_store.get_ref("develop")
        assert branch_hash == commit1.compute_hash()

    def test_create_branch_from_specific_commit(self, repo_memory, block_store):
        """Test creating a branch from a specific commit."""
        # Create two commits
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root1 = tree._hash_node(tree.root)
        commit1 = repo_memory.commit(tree_root1, "First commit")

        tree.insert_batch([(b'key2', b'value2')])
        tree_root2 = tree._hash_node(tree.root)
        commit2 = repo_memory.commit(tree_root2, "Second commit")

        # Create branch from first commit
        repo_memory.create_branch("branch-from-first", from_commit=commit1.compute_hash())

        # Verify branch points to first commit
        branch_hash = repo_memory.commit_graph_store.get_ref("branch-from-first")
        assert branch_hash == commit1.compute_hash()

    def test_create_duplicate_branch(self, repo_memory, block_store):
        """Test that creating a duplicate branch raises error."""
        # Create initial commit
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root = tree._hash_node(tree.root)
        repo_memory.commit(tree_root, "Initial commit")

        # Create branch
        repo_memory.create_branch("develop")

        # Try to create it again
        with pytest.raises(ValueError, match="already exists"):
            repo_memory.create_branch("develop")

    def test_checkout(self, repo_memory, block_store):
        """Test checking out a branch."""
        # Create initial commit and branch
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root = tree._hash_node(tree.root)
        repo_memory.commit(tree_root, "Initial commit")
        repo_memory.create_branch("develop")

        # Checkout develop
        repo_memory.checkout("develop")

        _, ref = repo_memory.get_head()
        assert ref == "develop"

    def test_checkout_nonexistent_branch(self, repo_memory):
        """Test that checking out nonexistent branch raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            repo_memory.checkout("nonexistent")

    def test_list_branches(self, repo_memory, block_store):
        """Test listing branches."""
        # Create initial commit and branches
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root = tree._hash_node(tree.root)
        commit = repo_memory.commit(tree_root, "Initial commit")
        repo_memory.create_branch("develop")
        repo_memory.create_branch("feature")

        branches = repo_memory.list_branches()
        assert "main" in branches
        assert "develop" in branches
        assert "feature" in branches
        assert len(branches) == 3

    def test_independent_branch_commits(self, repo_memory, block_store):
        """Test that commits on different branches are independent."""
        # Create initial commit
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root1 = tree._hash_node(tree.root)
        commit1 = repo_memory.commit(tree_root1, "Initial commit")

        # Create and checkout develop branch
        repo_memory.create_branch("develop")
        repo_memory.checkout("develop")

        # Make commit on develop
        tree.insert_batch([(b'key2', b'value2')])
        tree_root2 = tree._hash_node(tree.root)
        commit2 = repo_memory.commit(tree_root2, "Develop commit")

        # Verify develop is ahead
        develop_commit, _ = repo_memory.get_head()
        assert develop_commit.compute_hash() == commit2.compute_hash()

        # Checkout main and verify it's still at first commit
        repo_memory.checkout("main")
        main_commit, _ = repo_memory.get_head()
        assert main_commit.compute_hash() == commit1.compute_hash()

    def test_repo_with_sqlite_store(self, repo_sqlite, block_store):
        """Test that repo works with SQLite commit store."""
        # Create commits
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root1 = tree._hash_node(tree.root)
        commit1 = repo_sqlite.commit(tree_root1, "First commit")

        tree.insert_batch([(b'key2', b'value2')])
        tree_root2 = tree._hash_node(tree.root)
        commit2 = repo_sqlite.commit(tree_root2, "Second commit")

        # Verify commits are persisted
        head_commit, _ = repo_sqlite.get_head()
        assert head_commit.compute_hash() == commit2.compute_hash()

        # Verify parent relationship
        assert commit2.parents[0] == commit1.compute_hash()

    def test_integration_workflow(self, repo_memory, block_store):
        """Test a complete workflow: commits, branching, and tree operations."""
        # 1. Create initial tree and commit
        tree = ProllyTree(store=block_store)
        tree.insert_batch([
            (b'users/1', b'alice'),
            (b'users/2', b'bob'),
        ])
        tree_root1 = tree._hash_node(tree.root)
        commit1 = repo_memory.commit(tree_root1, "Add users")

        # 2. Create feature branch
        repo_memory.create_branch("feature/add-more-users")
        repo_memory.checkout("feature/add-more-users")

        # 3. Add more users on feature branch
        tree.insert_batch([
            (b'users/3', b'charlie'),
            (b'users/4', b'dave'),
        ])
        tree_root2 = tree._hash_node(tree.root)
        commit2 = repo_memory.commit(tree_root2, "Add more users")

        # 4. Verify feature branch has correct history
        feature_commit, ref = repo_memory.get_head()
        assert ref == "feature/add-more-users"
        assert feature_commit.parents[0] == commit1.compute_hash()

        # 5. Switch back to main
        repo_memory.checkout("main")
        main_commit, ref = repo_memory.get_head()
        assert ref == "main"
        assert main_commit.compute_hash() == commit1.compute_hash()

        # 6. Verify branches list
        branches = repo_memory.list_branches()
        assert len(branches) == 2
        assert "main" in branches
        assert "feature/add-more-users" in branches


class TestGetNodesToPush:
    """Tests for get_nodes_to_push functionality."""

    def test_push_all_nodes_no_base(self, repo_memory, block_store):
        """Test that with no base commit, all nodes are returned."""
        # Create a tree with multiple keys
        tree = ProllyTree(store=block_store)
        tree.insert_batch([
            (b'key1', b'value1'),
            (b'key2', b'value2'),
            (b'key3', b'value3'),
        ])
        tree_root = tree._hash_node(tree.root)
        block_store.put_node(tree_root, tree.root)
        commit1 = repo_memory.commit(tree_root, "First commit")

        # Get nodes to push with no base
        nodes = repo_memory.get_nodes_to_push(base_commit=None)

        # Should return all nodes in the tree
        assert len(nodes) > 0

    def test_incremental_push_single_key_change(self, repo_memory, block_store):
        """Test that changing a single key only pushes new/modified nodes."""
        # Create initial tree with many keys (use higher pattern for more nodes)
        tree = ProllyTree(store=block_store, pattern=0.1)  # Higher pattern = more nodes
        initial_keys = [(f'key{i:04d}'.encode(), f'value{i}'.encode()) for i in range(1000)]
        tree.insert_batch(initial_keys)
        tree_root1 = tree._hash_node(tree.root)
        block_store.put_node(tree_root1, tree.root)
        commit1 = repo_memory.commit(tree_root1, "Initial commit with 1000 keys")
        commit1_hash = commit1.compute_hash()

        # Get all nodes from first commit
        all_nodes = repo_memory.get_nodes_to_push(base_commit=None)
        initial_node_count = len(all_nodes)

        # Change a single key
        tree.insert_batch([(b'key0500', b'modified_value')])
        tree_root2 = tree._hash_node(tree.root)
        block_store.put_node(tree_root2, tree.root)
        commit2 = repo_memory.commit(tree_root2, "Modify one key")

        # Get nodes to push incrementally (only new since commit1)
        incremental_nodes = repo_memory.get_nodes_to_push(base_commit=commit1_hash)

        # The incremental push should be much smaller than the full push
        # A single key change should only affect a small number of nodes
        # (the leaf node and its ancestors up to the root)
        print(f"Initial nodes: {initial_node_count}, Incremental nodes: {len(incremental_nodes)}")

        # With 1000 keys and pattern 0.1, we should have ~100+ nodes
        # A single key change should affect < 20% of nodes
        assert len(incremental_nodes) < initial_node_count, \
            f"Incremental push ({len(incremental_nodes)}) should be smaller than full push ({initial_node_count})"

        assert len(incremental_nodes) < initial_node_count * 0.3, \
            f"Incremental push ({len(incremental_nodes)}) should be much smaller than full push ({initial_node_count})"

    def test_incremental_push_add_single_key(self, repo_memory, block_store):
        """Test that adding a single key only pushes new nodes."""
        # Create initial tree with many keys
        tree = ProllyTree(store=block_store, pattern=0.1)
        initial_keys = [(f'key{i:04d}'.encode(), f'value{i}'.encode()) for i in range(1000)]
        tree.insert_batch(initial_keys)
        tree_root1 = tree._hash_node(tree.root)
        block_store.put_node(tree_root1, tree.root)
        commit1 = repo_memory.commit(tree_root1, "Initial commit")
        commit1_hash = commit1.compute_hash()

        initial_node_count = len(repo_memory.get_nodes_to_push(base_commit=None))

        # Add a single new key
        tree.insert_batch([(b'new_key', b'new_value')])
        tree_root2 = tree._hash_node(tree.root)
        block_store.put_node(tree_root2, tree.root)
        commit2 = repo_memory.commit(tree_root2, "Add one key")

        # Get incremental nodes
        incremental_nodes = repo_memory.get_nodes_to_push(base_commit=commit1_hash)

        print(f"Initial nodes: {initial_node_count}, Incremental nodes after add: {len(incremental_nodes)}")

        assert len(incremental_nodes) < initial_node_count * 0.3, \
            f"Adding one key should not require pushing {len(incremental_nodes)} nodes (total: {initial_node_count})"

    def test_no_changes_no_nodes(self, repo_memory, block_store):
        """Test that if no changes were made, no nodes need to be pushed."""
        # Create a commit
        tree = ProllyTree(store=block_store)
        tree.insert_batch([(b'key1', b'value1')])
        tree_root = tree._hash_node(tree.root)
        block_store.put_node(tree_root, tree.root)
        commit1 = repo_memory.commit(tree_root, "First commit")
        commit1_hash = commit1.compute_hash()

        # Get nodes to push with the current commit as base
        # (i.e., nothing new since this commit)
        nodes = repo_memory.get_nodes_to_push(base_commit=commit1_hash)

        # Should return empty set since there are no new commits
        assert len(nodes) == 0, f"Expected 0 nodes, got {len(nodes)}"
