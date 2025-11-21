package prollycore

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Repo is a git-like repository abstraction
type Repo struct {
	BlockStore       BlockStore
	CommitGraphStore CommitGraphStore
	DefaultAuthor    string
}

// NewRepo creates a new repository
func NewRepo(blockStore BlockStore, commitStore CommitGraphStore, author string) *Repo {
	return &Repo{
		BlockStore:       blockStore,
		CommitGraphStore: commitStore,
		DefaultAuthor:    author,
	}
}

// InitEmpty initializes an empty repository
func (r *Repo) InitEmpty() Hash {
	// Create empty tree
	tree := NewProllyTree(r.BlockStore, 1.0/float64(TargetFanout), 0)
	rootHash := tree.GetRootHash()
	r.BlockStore.PutNode(rootHash, tree.Root)

	// Create initial commit
	commit := Commit{
		TreeRoot:  rootHash,
		Parents:   nil,
		Message:   "Initial commit",
		Timestamp: float64(time.Now().Unix()),
		Author:    r.DefaultAuthor,
		Pattern:   tree.PatternFloat(),
		Seed:      tree.Seed,
	}
	commitHash := HashCommit(commit)
	r.CommitGraphStore.PutCommit(commitHash, commit)

	// Set up refs
	r.CommitGraphStore.SetRef("main", commitHash)
	r.CommitGraphStore.SetHead("main")

	return commitHash
}

// Commit creates a new commit
func (r *Repo) Commit(treeRoot Hash, message string, pattern float64, seed uint32) (Hash, error) {
	// Get parent from HEAD
	headRef := r.CommitGraphStore.GetHead()
	if headRef == "" {
		return Hash{}, fmt.Errorf("no HEAD")
	}

	parentHash := r.CommitGraphStore.GetRef(headRef)
	if parentHash == nil {
		return Hash{}, fmt.Errorf("ref %s not found", headRef)
	}

	// Create commit
	commit := Commit{
		TreeRoot:  treeRoot,
		Parents:   []Hash{*parentHash},
		Message:   message,
		Timestamp: float64(time.Now().Unix()),
		Author:    r.DefaultAuthor,
		Pattern:   pattern,
		Seed:      seed,
	}
	commitHash := HashCommit(commit)
	r.CommitGraphStore.PutCommit(commitHash, commit)

	// Update ref
	r.CommitGraphStore.SetRef(headRef, commitHash)

	return commitHash, nil
}

// CreateBranch creates a new branch
func (r *Repo) CreateBranch(name string, fromRef string) error {
	commitHash, err := r.ResolveRef(fromRef)
	if err != nil {
		return err
	}
	r.CommitGraphStore.SetRef(name, commitHash)
	return nil
}

// Checkout switches to a branch
func (r *Repo) Checkout(branchName string) error {
	if r.CommitGraphStore.GetRef(branchName) == nil {
		return fmt.Errorf("branch %s not found", branchName)
	}
	r.CommitGraphStore.SetHead(branchName)
	return nil
}

// GetHead returns the current HEAD reference
func (r *Repo) GetHead() string {
	return r.CommitGraphStore.GetHead()
}

// ListBranches returns all branches
func (r *Repo) ListBranches() map[string]Hash {
	return r.CommitGraphStore.ListRefs()
}

// Log returns commit history
func (r *Repo) Log(startRef string, maxCount int) ([]struct {
	Hash   Hash
	Commit Commit
}, error) {
	var result []struct {
		Hash   Hash
		Commit Commit
	}

	commitHash, err := r.ResolveRef(startRef)
	if err != nil {
		return nil, err
	}

	visited := make(map[Hash]bool)
	queue := []Hash{commitHash}

	for len(queue) > 0 && (maxCount <= 0 || len(result) < maxCount) {
		current := queue[0]
		queue = queue[1:]

		if visited[current] {
			continue
		}
		visited[current] = true

		commit := r.CommitGraphStore.GetCommit(current)
		if commit == nil {
			continue
		}

		result = append(result, struct {
			Hash   Hash
			Commit Commit
		}{current, *commit})

		queue = append(queue, commit.Parents...)
	}

	return result, nil
}

// ResolveRef resolves a reference to a commit hash
func (r *Repo) ResolveRef(ref string) (Hash, error) {
	// Handle HEAD~n syntax
	if strings.HasPrefix(ref, "HEAD") {
		headRef := r.CommitGraphStore.GetHead()
		if headRef == "" {
			return Hash{}, fmt.Errorf("no HEAD")
		}

		hash := r.CommitGraphStore.GetRef(headRef)
		if hash == nil {
			return Hash{}, fmt.Errorf("ref %s not found", headRef)
		}

		// Parse ~n
		rest := ref[4:]
		if rest == "" {
			return *hash, nil
		}

		if strings.HasPrefix(rest, "~") {
			n, err := strconv.Atoi(rest[1:])
			if err != nil {
				return Hash{}, fmt.Errorf("invalid ref: %s", ref)
			}

			current := *hash
			for i := 0; i < n; i++ {
				commit := r.CommitGraphStore.GetCommit(current)
				if commit == nil || len(commit.Parents) == 0 {
					return Hash{}, fmt.Errorf("cannot go back %d commits", n)
				}
				current = commit.Parents[0]
			}
			return current, nil
		}

		return Hash{}, fmt.Errorf("invalid ref: %s", ref)
	}

	// Try as branch name
	if hash := r.CommitGraphStore.GetRef(ref); hash != nil {
		return *hash, nil
	}

	// Try as commit prefix
	if hash := r.CommitGraphStore.FindCommitByPrefix(ref); hash != nil {
		return *hash, nil
	}

	return Hash{}, fmt.Errorf("ref not found: %s", ref)
}

// GetTreeAtRef gets the tree root at a given reference
func (r *Repo) GetTreeAtRef(ref string) (Hash, *Commit, error) {
	commitHash, err := r.ResolveRef(ref)
	if err != nil {
		return Hash{}, nil, err
	}

	commit := r.CommitGraphStore.GetCommit(commitHash)
	if commit == nil {
		return Hash{}, nil, fmt.Errorf("commit not found")
	}

	return commit.TreeRoot, commit, nil
}

// GetReachableTreeRoots returns all tree roots reachable from commits
func (r *Repo) GetReachableTreeRoots() []Hash {
	var roots []Hash
	seen := make(map[Hash]bool)

	for _, commitHash := range r.CommitGraphStore.ListCommits() {
		commit := r.CommitGraphStore.GetCommit(commitHash)
		if commit != nil && !seen[commit.TreeRoot] {
			seen[commit.TreeRoot] = true
			roots = append(roots, commit.TreeRoot)
		}
	}

	return roots
}
