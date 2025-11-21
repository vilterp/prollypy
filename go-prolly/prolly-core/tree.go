package prollycore

import (
	"bytes"
	"math"
	"sort"

	"github.com/cespare/xxhash/v2"
)

const (
	MinNodeSize  = 2
	TargetFanout = 32
	MinFanout    = 8
)

// BatchStats tracks statistics for batch operations
type BatchStats struct {
	NodesCreated    int
	LeavesCreated   int
	InternalsCreated int
	NodesReused     int
	SubtreesReused  int
	NodesRead       int
}

// ProllyTree is a probabilistic B-tree
type ProllyTree struct {
	Pattern uint32
	Seed    uint32
	Store   BlockStore
	Root    *Node
	Stats   BatchStats
}

// NewProllyTree creates a new ProllyTree
func NewProllyTree(store BlockStore, pattern float64, seed uint32) *ProllyTree {
	return &ProllyTree{
		Pattern: uint32(pattern * float64(math.MaxUint32)),
		Seed:    seed,
		Store:   store,
		Root:    NewLeafNode(nil, nil),
	}
}

// NewProllyTreeFromRoot creates a ProllyTree from an existing root
func NewProllyTreeFromRoot(store BlockStore, root *Node, pattern float64, seed uint32) *ProllyTree {
	return &ProllyTree{
		Pattern: uint32(pattern * float64(math.MaxUint32)),
		Seed:    seed,
		Store:   store,
		Root:    root,
	}
}

// RollingHash computes a rolling hash using xxHash
func RollingHash(seed uint32, data []byte) uint32 {
	// Combine seed with data for rolling hash
	combined := make([]byte, 4+len(data))
	combined[0] = byte(seed)
	combined[1] = byte(seed >> 8)
	combined[2] = byte(seed >> 16)
	combined[3] = byte(seed >> 24)
	copy(combined[4:], data)
	h := xxhash.Sum64(combined)
	return uint32(h)
}

// Get retrieves a value by key
func (t *ProllyTree) Get(key []byte) []byte {
	return t.getFromNode(t.Root, key)
}

func (t *ProllyTree) getFromNode(node *Node, key []byte) []byte {
	if node.IsLeaf {
		idx := sort.Search(len(node.Keys), func(i int) bool {
			return bytes.Compare(node.Keys[i], key) >= 0
		})
		if idx < len(node.Keys) && bytes.Equal(node.Keys[idx], key) {
			return node.Values[idx]
		}
		return nil
	}

	// Find child using separators
	childIdx := len(node.Values) - 1
	for i, sep := range node.Keys {
		if bytes.Compare(key, sep) < 0 {
			childIdx = i
			break
		}
	}

	childHash := HashFromBytes(node.Values[childIdx])
	child := t.Store.GetNode(childHash)
	if child == nil {
		return nil
	}
	t.Stats.NodesRead++
	return t.getFromNode(child, key)
}

// InsertBatch inserts multiple key-value pairs
func (t *ProllyTree) InsertBatch(mutations []struct{ Key, Value []byte }) BatchStats {
	t.Stats = BatchStats{}

	if len(mutations) == 0 {
		return t.Stats
	}

	// Sort mutations
	sort.Slice(mutations, func(i, j int) bool {
		return bytes.Compare(mutations[i].Key, mutations[j].Key) < 0
	})

	// Convert to internal format
	muts := make([][2][]byte, len(mutations))
	for i, m := range mutations {
		muts[i] = [2][]byte{m.Key, m.Value}
	}

	// Rebuild tree
	newRoot := t.rebuildWithMutations(t.Root, muts)
	t.Root = newRoot

	// Store root
	rootHash := HashNode(t.Root)
	t.Store.PutNode(rootHash, t.Root)

	return t.Stats
}

func (t *ProllyTree) rebuildWithMutations(node *Node, mutations [][2][]byte) *Node {
	if len(mutations) == 0 {
		return node
	}

	if node.IsLeaf {
		return t.rebuildLeaf(node, mutations)
	}

	return t.rebuildInternal(node, mutations)
}

func (t *ProllyTree) rebuildLeaf(node *Node, mutations [][2][]byte) *Node {
	// Merge existing data with mutations
	merged := t.mergeLeafData(node, mutations)

	// Build new leaves with splitting
	leaves := t.buildLeaves(merged)

	if len(leaves) == 1 {
		return leaves[0]
	}

	// Build internal nodes from leaves
	return t.buildInternalFromChildren(leaves)
}

func (t *ProllyTree) mergeLeafData(node *Node, mutations [][2][]byte) [][2][]byte {
	// Create map from existing data
	existing := make(map[string][]byte)
	for i, k := range node.Keys {
		existing[string(k)] = node.Values[i]
	}

	// Apply mutations
	for _, m := range mutations {
		if len(m[1]) == 0 {
			delete(existing, string(m[0]))
		} else {
			existing[string(m[0])] = m[1]
		}
	}

	// Convert to sorted slice
	result := make([][2][]byte, 0, len(existing))
	for k, v := range existing {
		result = append(result, [2][]byte{[]byte(k), v})
	}
	sort.Slice(result, func(i, j int) bool {
		return bytes.Compare(result[i][0], result[j][0]) < 0
	})

	return result
}

func (t *ProllyTree) buildLeaves(data [][2][]byte) []*Node {
	if len(data) == 0 {
		return []*Node{NewLeafNode(nil, nil)}
	}

	var leaves []*Node
	var currentKeys, currentValues [][]byte
	var hashAcc uint32 = t.Seed

	for i, kv := range data {
		currentKeys = append(currentKeys, kv[0])
		currentValues = append(currentValues, kv[1])

		// Compute rolling hash
		hashAcc = RollingHash(hashAcc, kv[0])
		hashAcc = RollingHash(hashAcc, kv[1])

		// Check if we should split
		shouldSplit := hashAcc < t.Pattern && len(currentKeys) >= MinNodeSize
		isLast := i == len(data)-1

		if shouldSplit || isLast {
			if len(currentKeys) > 0 {
				leaf := NewLeafNode(currentKeys, currentValues)
				leafHash := HashNode(leaf)
				t.Store.PutNode(leafHash, leaf)
				leaves = append(leaves, leaf)
				t.Stats.LeavesCreated++
				t.Stats.NodesCreated++
			}
			currentKeys = nil
			currentValues = nil
			hashAcc = t.Seed
		}
	}

	return leaves
}

func (t *ProllyTree) rebuildInternal(node *Node, mutations [][2][]byte) *Node {
	// Partition mutations to children
	partitioned := t.partitionMutations(node, mutations)

	// Rebuild each child
	var newChildren []*Node
	for i, childHashBytes := range node.Values {
		childHash := HashFromBytes(childHashBytes)
		child := t.Store.GetNode(childHash)
		if child == nil {
			continue
		}
		t.Stats.NodesRead++

		childMuts := partitioned[i]
		if len(childMuts) == 0 {
			// Reuse subtree
			newChildren = append(newChildren, child)
			t.Stats.SubtreesReused++
		} else {
			rebuilt := t.rebuildWithMutations(child, childMuts)
			newChildren = append(newChildren, rebuilt)
		}
	}

	return t.buildInternalFromChildren(newChildren)
}

func (t *ProllyTree) partitionMutations(node *Node, mutations [][2][]byte) [][][2][]byte {
	partitioned := make([][][2][]byte, len(node.Values))

	for _, m := range mutations {
		// Find which child this mutation belongs to
		childIdx := len(node.Values) - 1
		for i, sep := range node.Keys {
			if bytes.Compare(m[0], sep) < 0 {
				childIdx = i
				break
			}
		}
		partitioned[childIdx] = append(partitioned[childIdx], m)
	}

	return partitioned
}

func (t *ProllyTree) buildInternalFromChildren(children []*Node) *Node {
	if len(children) == 0 {
		return NewLeafNode(nil, nil)
	}
	if len(children) == 1 {
		return children[0]
	}

	// Group children into internal nodes
	var nodes []*Node
	var currentChildren []*Node
	var hashAcc uint32 = t.Seed

	for i, child := range children {
		currentChildren = append(currentChildren, child)

		childHash := HashNode(child)
		hashAcc = RollingHash(hashAcc, childHash[:])

		shouldSplit := hashAcc < t.Pattern && len(currentChildren) >= MinFanout
		isLast := i == len(children)-1

		if shouldSplit || isLast {
			if len(currentChildren) > 0 {
				node := t.createInternalNode(currentChildren)
				nodes = append(nodes, node)
			}
			currentChildren = nil
			hashAcc = t.Seed
		}
	}

	if len(nodes) == 1 {
		return nodes[0]
	}

	return t.buildInternalFromChildren(nodes)
}

func (t *ProllyTree) createInternalNode(children []*Node) *Node {
	// Create separators and child hashes
	var keys [][]byte
	var values [][]byte

	for i, child := range children {
		childHash := HashNode(child)
		values = append(values, childHash[:])

		// Separator is first key of child (except for first child)
		if i > 0 {
			firstKey := getFirstKey(child, t.Store)
			keys = append(keys, firstKey)
		}
	}

	node := NewInternalNode(keys, values)
	nodeHash := HashNode(node)
	t.Store.PutNode(nodeHash, node)
	t.Stats.InternalsCreated++
	t.Stats.NodesCreated++

	return node
}

// GetRootHash returns the hash of the root node
func (t *ProllyTree) GetRootHash() Hash {
	return HashNode(t.Root)
}

// PatternFloat returns the pattern as a float64
func (t *ProllyTree) PatternFloat() float64 {
	return float64(t.Pattern) / float64(math.MaxUint32)
}
