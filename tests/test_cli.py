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
from contextlib import contextmanager

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


@contextmanager
def capture_stdout():
    """Context manager to capture stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout


class TestBranchCommand:
    """Tests for the branch CLI command."""

    def test_list_branches_initial_repo(self, repo_dir):
        """Test listing branches in a newly initialized repo."""
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_dir)
            result = output.getvalue()

        assert "* main" in result
        assert result.count("*") == 1  # Only one branch should be marked

    def test_create_branch(self, repo_dir):
        """Test creating a new branch."""
        with capture_stdout() as output:
            list_and_create_branch(name="feature-x", prolly_dir=repo_dir)
            result = output.getvalue()

        assert "Switched to a new branch 'feature-x'" in result
        assert "Initial commit" in result

        # Verify branch exists and is checked out
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_dir)
            list_output = output.getvalue()

        assert "* feature-x" in list_output
        assert "  main" in list_output

    def test_create_branch_from_specific_ref(self, repo_with_commits):
        """Test creating a branch from a specific ref."""
        # Create feature branch from main
        with capture_stdout() as output:
            list_and_create_branch(
                name="feature-y",
                from_ref="main",
                prolly_dir=repo_with_commits
            )
            result = output.getvalue()

        assert "Switched to a new branch 'feature-y'" in result
        assert "Add some data" in result  # Should show the commit message

        # Verify the new branch is checked out
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* feature-y" in list_output

    def test_create_duplicate_branch_error(self, repo_dir):
        """Test that creating a duplicate branch shows an error."""
        # Try to create 'main' which already exists
        with capture_stdout() as output:
            list_and_create_branch(name="main", prolly_dir=repo_dir)
            result = output.getvalue()

        assert "Error:" in result
        assert "already exists" in result

    def test_create_branch_from_nonexistent_ref(self, repo_dir):
        """Test creating a branch from a non-existent ref shows error."""
        with capture_stdout() as output:
            list_and_create_branch(
                name="new-branch",
                from_ref="nonexistent",
                prolly_dir=repo_dir
            )
            result = output.getvalue()

        assert "Error:" in result
        assert "not found" in result

    def test_list_multiple_branches(self, repo_with_commits):
        """Test listing when multiple branches exist."""
        # Create some branches (each creates and checks out)
        list_and_create_branch(name="develop", prolly_dir=repo_with_commits)
        list_and_create_branch(name="feature-1", prolly_dir=repo_with_commits)
        list_and_create_branch(name="feature-2", prolly_dir=repo_with_commits)

        # List all branches (currently on feature-2)
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            result = output.getvalue()

        # Verify all branches appear
        assert "develop" in result
        assert "feature-1" in result
        assert "* feature-2" in result  # Last created branch is checked out
        assert "  main" in result

        # Verify branches are sorted
        lines = [line.strip() for line in result.strip().split('\n')]
        branch_names = [line.replace("* ", "").replace("  ", "") for line in lines]
        assert branch_names == sorted(branch_names)

    def test_list_branches_shows_current_head(self, repo_with_commits):
        """Test that current HEAD is marked with * in branch list."""
        # Create a new branch (automatically checks it out)
        list_and_create_branch(name="feature", prolly_dir=repo_with_commits)

        # List branches
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            result = output.getvalue()

        assert "* feature" in result
        assert "  main" in result  # main should not have *


class TestCheckoutCommand:
    """Tests for the checkout CLI command."""

    def test_checkout_existing_branch(self, repo_with_commits):
        """Test checking out an existing branch."""
        # Create a new branch
        list_and_create_branch(name="develop", prolly_dir=repo_with_commits)

        # Checkout the new branch
        with capture_stdout() as output:
            checkout_branch(ref_name="develop", prolly_dir=repo_with_commits)
            result = output.getvalue()

        assert "Switched to branch 'develop'" in result
        assert "HEAD is now at" in result

        # Verify HEAD changed by listing branches
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* develop" in list_output

    def test_checkout_nonexistent_branch(self, repo_dir):
        """Test checking out a non-existent branch shows error."""
        with capture_stdout() as output:
            checkout_branch(ref_name="nonexistent", prolly_dir=repo_dir)
            result = output.getvalue()

        assert "Error:" in result
        assert "does not exist" in result

    def test_checkout_back_to_main(self, repo_with_commits):
        """Test switching branches and returning to main."""
        # Create and checkout feature branch
        list_and_create_branch(name="feature", prolly_dir=repo_with_commits)
        checkout_branch("feature", prolly_dir=repo_with_commits)

        # Verify we're on feature
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* feature" in list_output

        # Checkout main
        with capture_stdout() as output:
            checkout_branch(ref_name="main", prolly_dir=repo_with_commits)
            result = output.getvalue()

        assert "Switched to branch 'main'" in result

        # Verify we're back on main
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* main" in list_output

    def test_checkout_shows_commit_message(self, repo_with_commits):
        """Test that checkout shows the commit message of the checked out branch."""
        # Create a branch
        list_and_create_branch(name="test-branch", prolly_dir=repo_with_commits)

        # Checkout the branch
        with capture_stdout() as output:
            checkout_branch(ref_name="test-branch", prolly_dir=repo_with_commits)
            result = output.getvalue()

        assert "Switched to branch 'test-branch'" in result
        assert "HEAD is now at" in result
        # The commit message should be shown
        assert "Add some data" in result or "Initial commit" in result


class TestBranchAndCheckoutIntegration:
    """Integration tests for branch and checkout commands together."""

    def test_create_and_checkout_workflow(self, repo_with_commits):
        """Test a complete workflow of creating and checking out branches."""
        # Initial state: on main
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* main" in list_output

        # Create feature branch (automatically checks it out)
        with capture_stdout() as output:
            list_and_create_branch(name="feature", prolly_dir=repo_with_commits)
            create_output = output.getvalue()

        assert "Switched to a new branch 'feature'" in create_output

        # Now on feature
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* feature" in list_output

        # Switch back to main
        with capture_stdout() as output:
            checkout_branch(ref_name="main", prolly_dir=repo_with_commits)
            checkout_output = output.getvalue()

        assert "Switched to branch 'main'" in checkout_output

        # Verify we're back on main
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* main" in list_output

    def test_branch_independence(self, repo_with_commits):
        """Test that branches maintain independence after creation."""
        # Create develop branch from main (automatically checks it out)
        with capture_stdout() as output:
            list_and_create_branch(name="develop", prolly_dir=repo_with_commits)
            develop_output = output.getvalue()

        # Create feature branch from main (not from develop, automatically checks it out)
        with capture_stdout() as output:
            list_and_create_branch(name="feature", from_ref="main", prolly_dir=repo_with_commits)
            feature_output = output.getvalue()

        # Both should have been created from the same commit (main)
        # This is verified by them both showing the same commit message
        assert "Add some data" in develop_output or "Initial commit" in develop_output
        assert "Add some data" in feature_output or "Initial commit" in feature_output

        # Verify both branches exist
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "develop" in list_output
        assert "* feature" in list_output  # Currently on feature

    def test_multiple_branch_switches(self, repo_with_commits):
        """Test switching between multiple branches."""
        # Create multiple branches
        branches = ["branch-a", "branch-b", "branch-c"]
        for branch in branches:
            list_and_create_branch(name=branch, prolly_dir=repo_with_commits)

        # Switch to each branch and verify
        for branch in branches:
            checkout_branch(branch, prolly_dir=repo_with_commits)
            with capture_stdout() as output:
                list_and_create_branch(prolly_dir=repo_with_commits)
                list_output = output.getvalue()

            assert f"* {branch}" in list_output

        # Finally checkout main
        checkout_branch("main", prolly_dir=repo_with_commits)
        with capture_stdout() as output:
            list_and_create_branch(prolly_dir=repo_with_commits)
            list_output = output.getvalue()

        assert "* main" in list_output
