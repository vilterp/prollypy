package prollycore

import (
	"os"
	"path/filepath"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
)

// BlockStore is the interface for storing tree nodes
type BlockStore interface {
	PutNode(hash Hash, node *Node)
	GetNode(hash Hash) *Node
	DeleteNode(hash Hash) bool
	ListNodes() []Hash
	CountNodes() int
}

// MemoryBlockStore is an in-memory block store
type MemoryBlockStore struct {
	mu    sync.RWMutex
	nodes map[Hash]*Node
}

// NewMemoryBlockStore creates a new in-memory block store
func NewMemoryBlockStore() *MemoryBlockStore {
	return &MemoryBlockStore{
		nodes: make(map[Hash]*Node),
	}
}

func (s *MemoryBlockStore) PutNode(hash Hash, node *Node) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nodes[hash] = node
}

func (s *MemoryBlockStore) GetNode(hash Hash) *Node {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.nodes[hash]
}

func (s *MemoryBlockStore) DeleteNode(hash Hash) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.nodes[hash]; ok {
		delete(s.nodes, hash)
		return true
	}
	return false
}

func (s *MemoryBlockStore) ListNodes() []Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()
	hashes := make([]Hash, 0, len(s.nodes))
	for h := range s.nodes {
		hashes = append(hashes, h)
	}
	return hashes
}

func (s *MemoryBlockStore) CountNodes() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.nodes)
}

// FileSystemBlockStore stores nodes on the filesystem
type FileSystemBlockStore struct {
	basePath string
	mu       sync.RWMutex
}

// NewFileSystemBlockStore creates a new filesystem block store
func NewFileSystemBlockStore(basePath string) (*FileSystemBlockStore, error) {
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, err
	}
	return &FileSystemBlockStore{basePath: basePath}, nil
}

func (s *FileSystemBlockStore) nodePath(hash Hash) string {
	hexHash := hash.Hex()
	subdir := hexHash[:2]
	return filepath.Join(s.basePath, subdir, hexHash)
}

func (s *FileSystemBlockStore) PutNode(hash Hash, node *Node) {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := s.nodePath(hash)
	dir := filepath.Dir(path)
	os.MkdirAll(dir, 0755)

	data, err := node.Serialize()
	if err != nil {
		return
	}
	os.WriteFile(path, data, 0644)
}

func (s *FileSystemBlockStore) GetNode(hash Hash) *Node {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path := s.nodePath(hash)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	node, err := DeserializeNode(data)
	if err != nil {
		return nil
	}
	return node
}

func (s *FileSystemBlockStore) DeleteNode(hash Hash) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := s.nodePath(hash)
	err := os.Remove(path)
	return err == nil
}

func (s *FileSystemBlockStore) ListNodes() []Hash {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var hashes []Hash
	filepath.Walk(s.basePath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		name := info.Name()
		if len(name) == 32 { // hex encoded 16-byte hash
			if h, err := HashFromHex(name); err == nil {
				hashes = append(hashes, h)
			}
		}
		return nil
	})
	return hashes
}

func (s *FileSystemBlockStore) CountNodes() int {
	return len(s.ListNodes())
}

// CachedFSBlockStore is a filesystem store with LRU cache
type CachedFSBlockStore struct {
	fs    *FileSystemBlockStore
	cache *lru.Cache[Hash, *Node]
	mu    sync.RWMutex
}

// NewCachedFSBlockStore creates a cached filesystem block store
func NewCachedFSBlockStore(basePath string, cacheSize int) (*CachedFSBlockStore, error) {
	fs, err := NewFileSystemBlockStore(basePath)
	if err != nil {
		return nil, err
	}
	cache, err := lru.New[Hash, *Node](cacheSize)
	if err != nil {
		return nil, err
	}
	return &CachedFSBlockStore{fs: fs, cache: cache}, nil
}

func (s *CachedFSBlockStore) PutNode(hash Hash, node *Node) {
	s.mu.Lock()
	s.cache.Add(hash, node)
	s.mu.Unlock()
	s.fs.PutNode(hash, node)
}

func (s *CachedFSBlockStore) GetNode(hash Hash) *Node {
	s.mu.RLock()
	if node, ok := s.cache.Get(hash); ok {
		s.mu.RUnlock()
		return node
	}
	s.mu.RUnlock()

	node := s.fs.GetNode(hash)
	if node != nil {
		s.mu.Lock()
		s.cache.Add(hash, node)
		s.mu.Unlock()
	}
	return node
}

func (s *CachedFSBlockStore) DeleteNode(hash Hash) bool {
	s.mu.Lock()
	s.cache.Remove(hash)
	s.mu.Unlock()
	return s.fs.DeleteNode(hash)
}

func (s *CachedFSBlockStore) ListNodes() []Hash {
	return s.fs.ListNodes()
}

func (s *CachedFSBlockStore) CountNodes() int {
	return s.fs.CountNodes()
}
