package prollycore

import (
	"strings"
	"sync"
)

// Commit represents a commit in the graph
type Commit struct {
	TreeRoot  Hash     `msgpack:"tree_root"`
	Parents   []Hash   `msgpack:"parents"`
	Message   string   `msgpack:"message"`
	Timestamp float64  `msgpack:"timestamp"`
	Author    string   `msgpack:"author"`
	Pattern   float64  `msgpack:"pattern"`
	Seed      uint32   `msgpack:"seed"`
}

// CommitGraphStore is the interface for storing commits
type CommitGraphStore interface {
	PutCommit(hash Hash, commit Commit)
	GetCommit(hash Hash) *Commit
	GetParents(hash Hash) []Hash
	SetRef(name string, hash Hash)
	GetRef(name string) *Hash
	ListRefs() map[string]Hash
	SetHead(refName string)
	GetHead() string
	FindCommitByPrefix(prefix string) *Hash
	ListCommits() []Hash
}

// MemoryCommitGraphStore is an in-memory implementation
type MemoryCommitGraphStore struct {
	mu      sync.RWMutex
	commits map[Hash]Commit
	refs    map[string]Hash
	head    string
}

// NewMemoryCommitGraphStore creates a new in-memory commit store
func NewMemoryCommitGraphStore() *MemoryCommitGraphStore {
	return &MemoryCommitGraphStore{
		commits: make(map[Hash]Commit),
		refs:    make(map[string]Hash),
	}
}

func (s *MemoryCommitGraphStore) PutCommit(hash Hash, commit Commit) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.commits[hash] = commit
}

func (s *MemoryCommitGraphStore) GetCommit(hash Hash) *Commit {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if c, ok := s.commits[hash]; ok {
		return &c
	}
	return nil
}

func (s *MemoryCommitGraphStore) GetParents(hash Hash) []Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if c, ok := s.commits[hash]; ok {
		return c.Parents
	}
	return nil
}

func (s *MemoryCommitGraphStore) SetRef(name string, hash Hash) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refs[name] = hash
}

func (s *MemoryCommitGraphStore) GetRef(name string) *Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if h, ok := s.refs[name]; ok {
		return &h
	}
	return nil
}

func (s *MemoryCommitGraphStore) ListRefs() map[string]Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make(map[string]Hash)
	for k, v := range s.refs {
		result[k] = v
	}
	return result
}

func (s *MemoryCommitGraphStore) SetHead(refName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.head = refName
}

func (s *MemoryCommitGraphStore) GetHead() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.head
}

func (s *MemoryCommitGraphStore) FindCommitByPrefix(prefix string) *Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for h := range s.commits {
		if strings.HasPrefix(h.Hex(), prefix) {
			return &h
		}
	}
	return nil
}

func (s *MemoryCommitGraphStore) ListCommits() []Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]Hash, 0, len(s.commits))
	for h := range s.commits {
		result = append(result, h)
	}
	return result
}

// HashCommit computes the hash of a commit
func HashCommit(c Commit) Hash {
	var buf []byte
	buf = append(buf, c.TreeRoot[:]...)
	for _, p := range c.Parents {
		buf = append(buf, p[:]...)
	}
	buf = append(buf, []byte(c.Message)...)
	buf = append(buf, []byte(c.Author)...)
	return ComputeHash(buf)
}
