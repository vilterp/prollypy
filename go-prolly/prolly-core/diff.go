package prollycore

import (
	"bytes"
)

// DiffEventType represents the type of diff event
type DiffEventType int

const (
	DiffAdded DiffEventType = iota
	DiffDeleted
	DiffModified
)

// DiffEvent represents a single diff event
type DiffEvent struct {
	Type     DiffEventType
	Key      []byte
	Value    []byte
	OldValue []byte
}

// DiffStats tracks diff algorithm statistics
type DiffStats struct {
	SubtreesSkipped int
	NodesCompared   int
}

// Diff computes the difference between two trees
func Diff(store BlockStore, oldRoot, newRoot Hash, prefix []byte) ([]DiffEvent, DiffStats) {
	var events []DiffEvent
	var stats DiffStats

	oldCursor := NewTreeCursorWithSeek(store, oldRoot, prefix)
	newCursor := NewTreeCursorWithSeek(store, newRoot, prefix)

	for !oldCursor.Done() || !newCursor.Done() {
		oldKey, oldValue := oldCursor.Current()
		newKey, newValue := newCursor.Current()

		// Check prefix bounds
		if prefix != nil {
			if oldKey != nil && !bytes.HasPrefix(oldKey, prefix) {
				oldKey, oldValue = nil, nil
			}
			if newKey != nil && !bytes.HasPrefix(newKey, prefix) {
				newKey, newValue = nil, nil
			}
		}

		if oldKey == nil && newKey == nil {
			break
		}

		stats.NodesCompared++

		// Try to skip identical subtrees
		oldHash := oldCursor.PeekNextHash()
		newHash := newCursor.PeekNextHash()
		if !oldHash.IsEmpty() && oldHash == newHash {
			oldCursor.SkipSubtree()
			newCursor.SkipSubtree()
			stats.SubtreesSkipped++
			continue
		}

		if oldKey == nil {
			// Added
			events = append(events, DiffEvent{
				Type:  DiffAdded,
				Key:   newKey,
				Value: newValue,
			})
			newCursor.Next()
		} else if newKey == nil {
			// Deleted
			events = append(events, DiffEvent{
				Type:     DiffDeleted,
				Key:      oldKey,
				OldValue: oldValue,
			})
			oldCursor.Next()
		} else {
			cmp := bytes.Compare(oldKey, newKey)
			if cmp < 0 {
				// Deleted
				events = append(events, DiffEvent{
					Type:     DiffDeleted,
					Key:      oldKey,
					OldValue: oldValue,
				})
				oldCursor.Next()
			} else if cmp > 0 {
				// Added
				events = append(events, DiffEvent{
					Type:  DiffAdded,
					Key:   newKey,
					Value: newValue,
				})
				newCursor.Next()
			} else {
				// Same key - check value
				if !bytes.Equal(oldValue, newValue) {
					events = append(events, DiffEvent{
						Type:     DiffModified,
						Key:      oldKey,
						OldValue: oldValue,
						Value:    newValue,
					})
				}
				oldCursor.Next()
				newCursor.Next()
			}
		}
	}

	return events, stats
}

// DiffWithLimit computes diff with a limit on events
func DiffWithLimit(store BlockStore, oldRoot, newRoot Hash, prefix []byte, limit int) ([]DiffEvent, DiffStats) {
	events, stats := Diff(store, oldRoot, newRoot, prefix)
	if limit > 0 && len(events) > limit {
		events = events[:limit]
	}
	return events, stats
}
