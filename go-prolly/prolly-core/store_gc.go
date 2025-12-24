package prollycore

// GCStats tracks garbage collection statistics
type GCStats struct {
	TotalNodes     int
	ReachableNodes int
	GarbageNodes   int
}

// FindReachableNodes finds all nodes reachable from the given roots
func FindReachableNodes(store BlockStore, roots []Hash) map[Hash]bool {
	reachable := make(map[Hash]bool)
	stack := make([]Hash, len(roots))
	copy(stack, roots)

	for len(stack) > 0 {
		// Pop
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if reachable[current] || current.IsEmpty() {
			continue
		}
		reachable[current] = true

		node := store.GetNode(current)
		if node == nil {
			continue
		}

		if !node.IsLeaf {
			for _, childHashBytes := range node.Values {
				childHash := HashFromBytes(childHashBytes)
				if !reachable[childHash] {
					stack = append(stack, childHash)
				}
			}
		}
	}

	return reachable
}

// FindGarbageNodes finds nodes not reachable from any root
func FindGarbageNodes(store BlockStore, roots []Hash) []Hash {
	reachable := FindReachableNodes(store, roots)

	var garbage []Hash
	for _, hash := range store.ListNodes() {
		if !reachable[hash] {
			garbage = append(garbage, hash)
		}
	}

	return garbage
}

// GarbageCollect removes unreachable nodes
func GarbageCollect(store BlockStore, roots []Hash, dryRun bool) GCStats {
	totalNodes := store.CountNodes()
	garbage := FindGarbageNodes(store, roots)

	if !dryRun {
		for _, hash := range garbage {
			store.DeleteNode(hash)
		}
	}

	return GCStats{
		TotalNodes:     totalNodes,
		ReachableNodes: totalNodes - len(garbage),
		GarbageNodes:   len(garbage),
	}
}
