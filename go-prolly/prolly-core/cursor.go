package prollycore

import (
	"bytes"
)

// TreeCursor provides O(log n) tree traversal
type TreeCursor struct {
	store   BlockStore
	stack   []stackEntry
	current *struct {
		Key   []byte
		Value []byte
	}
}

type stackEntry struct {
	node  *Node
	index int
}

// NewTreeCursor creates a new cursor
func NewTreeCursor(store BlockStore, rootHash Hash) *TreeCursor {
	root := store.GetNode(rootHash)
	if root == nil {
		return &TreeCursor{store: store}
	}

	cursor := &TreeCursor{
		store: store,
		stack: []stackEntry{{node: root, index: 0}},
	}
	cursor.descend()
	return cursor
}

// NewTreeCursorWithSeek creates a cursor and seeks to the given key
func NewTreeCursorWithSeek(store BlockStore, rootHash Hash, seekTo []byte) *TreeCursor {
	root := store.GetNode(rootHash)
	if root == nil {
		return &TreeCursor{store: store}
	}

	cursor := &TreeCursor{
		store: store,
		stack: []stackEntry{{node: root, index: 0}},
	}

	if seekTo != nil {
		cursor.seek(seekTo)
	} else {
		cursor.descend()
	}

	return cursor
}

// seek moves the cursor to the first key >= target
func (c *TreeCursor) seek(target []byte) {
	if len(c.stack) == 0 {
		return
	}

	// Navigate down to the target
	for {
		entry := &c.stack[len(c.stack)-1]
		node := entry.node

		if node.IsLeaf {
			// Find first key >= target
			for i, key := range node.Keys {
				if bytes.Compare(key, target) >= 0 {
					entry.index = i
					c.setCurrent(key, node.Values[i])
					return
				}
			}
			// All keys < target, need to go to next leaf
			entry.index = len(node.Keys)
			c.advance()
			return
		}

		// Internal node: find child containing target
		childIdx := len(node.Values) - 1
		for i, sep := range node.Keys {
			if bytes.Compare(target, sep) < 0 {
				childIdx = i
				break
			}
		}

		entry.index = childIdx
		childHash := HashFromBytes(node.Values[childIdx])
		child := c.store.GetNode(childHash)
		if child == nil {
			c.current = nil
			return
		}
		c.stack = append(c.stack, stackEntry{node: child, index: 0})
	}
}

// descend moves to the leftmost leaf
func (c *TreeCursor) descend() {
	for len(c.stack) > 0 {
		entry := &c.stack[len(c.stack)-1]
		node := entry.node

		if node.IsLeaf {
			if len(node.Keys) > 0 {
				c.setCurrent(node.Keys[0], node.Values[0])
			} else {
				c.current = nil
			}
			return
		}

		// Go to first child
		if len(node.Values) == 0 {
			c.current = nil
			return
		}
		childHash := HashFromBytes(node.Values[0])
		child := c.store.GetNode(childHash)
		if child == nil {
			c.current = nil
			return
		}
		c.stack = append(c.stack, stackEntry{node: child, index: 0})
	}
	c.current = nil
}

func (c *TreeCursor) setCurrent(key, value []byte) {
	c.current = &struct {
		Key   []byte
		Value []byte
	}{key, value}
}

// Current returns the current key-value pair
func (c *TreeCursor) Current() ([]byte, []byte) {
	if c.current == nil {
		return nil, nil
	}
	return c.current.Key, c.current.Value
}

// Next advances the cursor and returns the next key-value pair
func (c *TreeCursor) Next() ([]byte, []byte) {
	if c.current == nil {
		return nil, nil
	}
	key, value := c.current.Key, c.current.Value
	c.advance()
	return key, value
}

// advance moves to the next entry
func (c *TreeCursor) advance() {
	if len(c.stack) == 0 {
		c.current = nil
		return
	}

	// Move to next entry in current leaf
	entry := &c.stack[len(c.stack)-1]
	entry.index++

	if entry.node.IsLeaf {
		if entry.index < len(entry.node.Keys) {
			c.setCurrent(entry.node.Keys[entry.index], entry.node.Values[entry.index])
			return
		}
	}

	// Need to go up and find next subtree
	for len(c.stack) > 0 {
		c.stack = c.stack[:len(c.stack)-1]
		if len(c.stack) == 0 {
			c.current = nil
			return
		}

		entry := &c.stack[len(c.stack)-1]
		entry.index++

		if entry.index < len(entry.node.Values) {
			// Go down to next child
			childHash := HashFromBytes(entry.node.Values[entry.index])
			child := c.store.GetNode(childHash)
			if child == nil {
				continue
			}
			c.stack = append(c.stack, stackEntry{node: child, index: 0})
			c.descend()
			return
		}
	}

	c.current = nil
}

// PeekNextHash returns the hash of the next subtree without advancing
func (c *TreeCursor) PeekNextHash() Hash {
	if len(c.stack) == 0 {
		return EmptyHash
	}
	entry := c.stack[len(c.stack)-1]
	if entry.node.IsLeaf {
		return EmptyHash
	}
	if entry.index >= len(entry.node.Values) {
		return EmptyHash
	}
	return HashFromBytes(entry.node.Values[entry.index])
}

// SkipSubtree skips the current subtree
func (c *TreeCursor) SkipSubtree() {
	if len(c.stack) == 0 {
		return
	}

	// Pop current position and advance
	c.stack = c.stack[:len(c.stack)-1]
	if len(c.stack) > 0 {
		c.advance()
	} else {
		c.current = nil
	}
}

// Done returns true if the cursor is exhausted
func (c *TreeCursor) Done() bool {
	return c.current == nil
}
