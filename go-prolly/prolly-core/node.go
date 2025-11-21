package prollycore

import (
	"bytes"
	"fmt"

	"github.com/vmihailenco/msgpack/v5"
)

// Node represents a tree node (leaf or internal)
type Node struct {
	IsLeaf bool       `msgpack:"is_leaf"`
	Keys   [][]byte   `msgpack:"keys"`
	Values [][]byte   `msgpack:"values"`
}

// NewLeafNode creates a new leaf node
func NewLeafNode(keys, values [][]byte) *Node {
	return &Node{
		IsLeaf: true,
		Keys:   keys,
		Values: values,
	}
}

// NewInternalNode creates a new internal node
func NewInternalNode(keys [][]byte, childHashes [][]byte) *Node {
	return &Node{
		IsLeaf: false,
		Keys:   keys,
		Values: childHashes,
	}
}

// NumChildren returns the number of children (for internal nodes)
func (n *Node) NumChildren() int {
	return len(n.Values)
}

// NumEntries returns the number of key-value entries (for leaf nodes)
func (n *Node) NumEntries() int {
	return len(n.Keys)
}

// Serialize encodes the node to bytes
func (n *Node) Serialize() ([]byte, error) {
	return msgpack.Marshal(n)
}

// DeserializeNode decodes a node from bytes
func DeserializeNode(data []byte) (*Node, error) {
	var node Node
	err := msgpack.Unmarshal(data, &node)
	if err != nil {
		return nil, err
	}
	return &node, nil
}

// HashNode computes the content hash of a node
func HashNode(n *Node) Hash {
	var buf bytes.Buffer
	if n.IsLeaf {
		for i := range n.Keys {
			buf.Write(n.Keys[i])
			buf.WriteByte(':')
			buf.Write(n.Values[i])
			if i < len(n.Keys)-1 {
				buf.WriteByte('|')
			}
		}
	} else {
		for i := range n.Values {
			if i < len(n.Keys) {
				buf.Write(n.Keys[i])
			}
			buf.WriteByte(':')
			buf.Write(n.Values[i])
			if i < len(n.Values)-1 {
				buf.WriteByte('|')
			}
		}
	}
	return ComputeHash(buf.Bytes())
}

// Validate checks the node's invariants recursively
func (n *Node) Validate(store BlockStore) error {
	if n.IsLeaf {
		// Check keys are sorted
		for i := 1; i < len(n.Keys); i++ {
			if bytes.Compare(n.Keys[i-1], n.Keys[i]) >= 0 {
				return fmt.Errorf("leaf keys not sorted at index %d", i)
			}
		}
		if len(n.Keys) != len(n.Values) {
			return fmt.Errorf("leaf keys/values length mismatch")
		}
	} else {
		// Internal node: n-1 separators for n children
		if len(n.Keys) != len(n.Values)-1 {
			return fmt.Errorf("internal node: expected %d separators, got %d", len(n.Values)-1, len(n.Keys))
		}
		// Check separators are sorted
		for i := 1; i < len(n.Keys); i++ {
			if bytes.Compare(n.Keys[i-1], n.Keys[i]) >= 0 {
				return fmt.Errorf("separators not sorted at index %d", i)
			}
		}
		// Validate children recursively
		for i, childHashBytes := range n.Values {
			childHash := HashFromBytes(childHashBytes)
			child := store.GetNode(childHash)
			if child == nil {
				return fmt.Errorf("child %d not found", i)
			}
			if err := child.Validate(store); err != nil {
				return fmt.Errorf("child %d: %w", i, err)
			}
			// Check separator invariant: keys[i] = first key in child i+1
			if i < len(n.Keys) {
				firstKey := getFirstKey(child, store)
				if !bytes.Equal(n.Keys[i], firstKey) {
					return fmt.Errorf("separator invariant violated at index %d", i)
				}
			}
		}
	}
	return nil
}

// getFirstKey returns the first key in the subtree
func getFirstKey(n *Node, store BlockStore) []byte {
	if n.IsLeaf {
		if len(n.Keys) == 0 {
			return nil
		}
		return n.Keys[0]
	}
	if len(n.Values) == 0 {
		return nil
	}
	childHash := HashFromBytes(n.Values[0])
	child := store.GetNode(childHash)
	if child == nil {
		return nil
	}
	return getFirstKey(child, store)
}

// CollectKeys returns all keys in the subtree in order
func CollectKeys(n *Node, store BlockStore) [][]byte {
	var keys [][]byte
	if n.IsLeaf {
		keys = append(keys, n.Keys...)
	} else {
		for _, childHashBytes := range n.Values {
			childHash := HashFromBytes(childHashBytes)
			child := store.GetNode(childHash)
			if child != nil {
				keys = append(keys, CollectKeys(child, store)...)
			}
		}
	}
	return keys
}
