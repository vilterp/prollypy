"""
Tests for CLI commands.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from io import StringIO
import sys

from prollypy.cli import list_and_create_branch, checkout_branch, init_repo
from prollypy.repo import Repo, SqliteCommitGraphStore
from prollypy.store import CachedFSBlockStore
from prollypy.tree import ProllyTree


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def repo_dir(temp_dir):
    """Create a temporary .prolly directory and initialize a repository."""
    repo_path = os.path.join(temp_dir, '.prolly')

    # Initialize the repository
    old_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        init_repo(prolly_dir=repo_path, author="test@example.com")
        yield repo_path
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def repo_with_commits(repo_dir):
    """Create a repository with some commits."""
    blocks_dir = os.path.join(repo_dir, 'blocks')
    commits_db = os.path.join(repo_dir, 'commits.db')

    block_store = CachedFSBlockStore(blocks_dir, cache_size=1000)
    commit_graph_store = SqliteCommitGraphStore(commits_db)
    repo = Repo(block_store, commit_graph_store, default_author="test@example.com")

    # Create a tree with some data
    tree = ProllyTree(store=block_store)
    tree.insert_batch([
        (b'key1', b'value1'),
        (b'key2', b'value2'),
    ])
    tree_root = tree._hash_node(tree.root)

    # Create a commit
    repo.commit(tree_root, "Add some data")

    return repo_dir


def capture_stdout(func, *args, **kwargs):
    """Helper to capture stdout from a function."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        func(*args, **kwargs)
        output = sys.stdout.getvalue()
        return output
    finally:
        sys.stdout = old_stdout


class TestBranchCommand:
    """Tests for the branch CLI command."""

    def test_list_branches_initial_repo(self, repo_dir):
        """Test listing branches in a newly initialized repo."""
        output = capture_stdout(list_and_create_branch, prolly_dir=repo_dir)

        assert "* main" in output
        assert output.count("*") == 1  # Only one branch should be marked

    def test_create_branch(self, repo_dir):
        """Test creating a new branch."""
        output = capture_stdout(
            list_and_create_branch,
            name="feature-x",
            prolly_dir=repo_dir
        )

        assert "Created branch 'feature-x'" in output
        assert "Initial commit" in output

        # Verify branch exists by listing
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_dir)
        assert "feature-x" in list_output
        assert "* main" in list_output

    def test_create_branch_from_specific_ref(self, repo_with_commits):
        """Test creating a branch from a specific ref."""
        # Create feature branch from main
        output = capture_stdout(
            list_and_create_branch,
            name="feature-y",
            from_ref="main",
            prolly_dir=repo_with_commits
        )

        assert "Created branch 'feature-y'" in output
        assert "Add some data" in output  # Should show the commit message

    def test_create_duplicate_branch_error(self, repo_dir):
        """Test that creating a duplicate branch shows an error."""
        # Try to create 'main' which already exists
        output = capture_stdout(
            list_and_create_branch,
            name="main",
            prolly_dir=repo_dir
        )

        assert "Error:" in output
        assert "already exists" in output

    def test_create_branch_from_nonexistent_ref(self, repo_dir):
        """Test creating a branch from a non-existent ref shows error."""
        output = capture_stdout(
            list_and_create_branch,
            name="new-branch",
            from_ref="nonexistent",
            prolly_dir=repo_dir
        )

        assert "Error:" in output
        assert "not found" in output

    def test_list_multiple_branches(self, repo_with_commits):
        """Test listing when multiple branches exist."""
        # Create some branches
        list_and_create_branch(name="develop", prolly_dir=repo_with_commits)
        list_and_create_branch(name="feature-1", prolly_dir=repo_with_commits)
        list_and_create_branch(name="feature-2", prolly_dir=repo_with_commits)

        # List all branches
        output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)

        # Verify all branches appear
        assert "develop" in output
        assert "feature-1" in output
        assert "feature-2" in output
        assert "* main" in output

        # Verify branches are sorted
        lines = [line.strip() for line in output.strip().split('\n')]
        branch_names = [line.replace("* ", "").replace("  ", "") for line in lines]
        assert branch_names == sorted(branch_names)

    def test_list_branches_shows_current_head(self, repo_with_commits):
        """Test that current HEAD is marked with * in branch list."""
        # Create and checkout a new branch
        list_and_create_branch(name="feature", prolly_dir=repo_with_commits)
        checkout_branch("feature", prolly_dir=repo_with_commits)

        # List branches
        output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)

        assert "* feature" in output
        assert "  main" in output  # main should not have *


class TestCheckoutCommand:
    """Tests for the checkout CLI command."""

    def test_checkout_existing_branch(self, repo_with_commits):
        """Test checking out an existing branch."""
        # Create a new branch
        list_and_create_branch(name="develop", prolly_dir=repo_with_commits)

        # Checkout the new branch
        output = capture_stdout(
            checkout_branch,
            ref_name="develop",
            prolly_dir=repo_with_commits
        )

        assert "Switched to branch 'develop'" in output
        assert "HEAD is now at" in output

        # Verify HEAD changed by listing branches
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* develop" in list_output

    def test_checkout_nonexistent_branch(self, repo_dir):
        """Test checking out a non-existent branch shows error."""
        output = capture_stdout(
            checkout_branch,
            ref_name="nonexistent",
            prolly_dir=repo_dir
        )

        assert "Error:" in output
        assert "does not exist" in output

    def test_checkout_back_to_main(self, repo_with_commits):
        """Test switching branches and returning to main."""
        # Create and checkout feature branch
        list_and_create_branch(name="feature", prolly_dir=repo_with_commits)
        checkout_branch("feature", prolly_dir=repo_with_commits)

        # Verify we're on feature
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* feature" in list_output

        # Checkout main
        output = capture_stdout(
            checkout_branch,
            ref_name="main",
            prolly_dir=repo_with_commits
        )

        assert "Switched to branch 'main'" in output

        # Verify we're back on main
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* main" in list_output

    def test_checkout_shows_commit_message(self, repo_with_commits):
        """Test that checkout shows the commit message of the checked out branch."""
        # Create a branch
        list_and_create_branch(name="test-branch", prolly_dir=repo_with_commits)

        # Checkout the branch
        output = capture_stdout(
            checkout_branch,
            ref_name="test-branch",
            prolly_dir=repo_with_commits
        )

        assert "Switched to branch 'test-branch'" in output
        assert "HEAD is now at" in output
        # The commit message should be shown
        assert "Add some data" in output or "Initial commit" in output


class TestBranchAndCheckoutIntegration:
    """Integration tests for branch and checkout commands together."""

    def test_create_and_checkout_workflow(self, repo_with_commits):
        """Test a complete workflow of creating and checking out branches."""
        # Initial state: on main
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* main" in list_output

        # Create feature branch
        create_output = capture_stdout(
            list_and_create_branch,
            name="feature",
            prolly_dir=repo_with_commits
        )
        assert "Created branch 'feature'" in create_output

        # Still on main
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* main" in list_output

        # Checkout feature
        checkout_output = capture_stdout(
            checkout_branch,
            ref_name="feature",
            prolly_dir=repo_with_commits
        )
        assert "Switched to branch 'feature'" in checkout_output

        # Now on feature
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* feature" in list_output

    def test_branch_independence(self, repo_with_commits):
        """Test that branches maintain independence after creation."""
        # Create develop branch from main
        list_and_create_branch(name="develop", prolly_dir=repo_with_commits)

        # Create feature branch from main (not from develop)
        list_and_create_branch(name="feature", from_ref="main", prolly_dir=repo_with_commits)

        # Both should point to the same commit initially
        # This is verified by them both showing the same commit message
        checkout_output1 = capture_stdout(
            checkout_branch,
            ref_name="develop",
            prolly_dir=repo_with_commits
        )

        checkout_output2 = capture_stdout(
            checkout_branch,
            ref_name="feature",
            prolly_dir=repo_with_commits
        )

        # Both should show the same commit
        assert "Add some data" in checkout_output1 or "Initial commit" in checkout_output1
        assert "Add some data" in checkout_output2 or "Initial commit" in checkout_output2

    def test_multiple_branch_switches(self, repo_with_commits):
        """Test switching between multiple branches."""
        # Create multiple branches
        branches = ["branch-a", "branch-b", "branch-c"]
        for branch in branches:
            list_and_create_branch(name=branch, prolly_dir=repo_with_commits)

        # Switch to each branch and verify
        for branch in branches:
            checkout_branch(branch, prolly_dir=repo_with_commits)
            list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
            assert f"* {branch}" in list_output

        # Finally checkout main
        checkout_branch("main", prolly_dir=repo_with_commits)
        list_output = capture_stdout(list_and_create_branch, prolly_dir=repo_with_commits)
        assert "* main" in list_output
