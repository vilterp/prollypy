"""
Commonality analysis for comparing two ProllyTree root hashes.

Computes a Venn diagram showing which nodes are unique to each tree
and which are shared between them.
"""

from typing import Set, Dict, Any
from .store import BlockStore


def collect_node_hashes(store: BlockStore, root_hash: bytes) -> Set[bytes]:
    """
    Recursively collect all node hashes in a tree.

    Args:
        store: Storage backend to retrieve nodes
        root_hash: Hash of the root node to start from

    Returns:
        Set of all node hashes in the tree (including root)
    """
    visited: Set[bytes] = set()
    to_visit = [root_hash]

    while to_visit:
        node_hash = to_visit.pop()

        # Skip if already visited
        if node_hash in visited:
            continue

        visited.add(node_hash)

        # Get the node
        node = store.get_node(node_hash)
        if node is None:
            raise ValueError(f"Node {node_hash.hex()} not found in store")

        # If internal node, add children to visit list
        if not node.is_leaf:
            for child_hash in node.values:
                if child_hash not in visited:
                    to_visit.append(child_hash)

    return visited


def compute_commonality(store: BlockStore, left_hash: bytes, right_hash: bytes) -> Dict[str, Any]:
    """
    Compute commonality between two trees (Venn diagram analysis).

    Args:
        store: Storage backend containing both trees
        left_hash: Root hash of left tree (as bytes)
        right_hash: Root hash of right tree (as bytes)

    Returns:
        Dictionary with:
            - left_only: Set of hashes only in left tree
            - right_only: Set of hashes only in right tree
            - both: Set of hashes in both trees
            - left_total: Total nodes in left tree
            - right_total: Total nodes in right tree
            - both_count: Number of shared nodes
    """
    # Collect all node hashes from both trees
    left_nodes = collect_node_hashes(store, left_hash)
    right_nodes = collect_node_hashes(store, right_hash)

    # Compute set operations
    both = left_nodes & right_nodes
    left_only = left_nodes - right_nodes
    right_only = right_nodes - left_nodes

    return {
        'left_only': left_only,
        'right_only': right_only,
        'both': both,
        'left_total': len(left_nodes),
        'right_total': len(right_nodes),
        'both_count': len(both),
    }


def print_commonality_report(left_hash: str, right_hash: str, stats: Dict[str, Any]):
    """
    Print a formatted commonality report.

    Args:
        left_hash: Root hash of left tree
        right_hash: Root hash of right tree
        stats: Statistics from compute_commonality()
    """
    left_total = stats['left_total']
    right_total = stats['right_total']
    both_count = stats['both_count']
    left_only_count = len(stats['left_only'])
    right_only_count = len(stats['right_only'])

    # Calculate percentages
    left_only_pct = (left_only_count / left_total * 100) if left_total > 0 else 0
    right_only_pct = (right_only_count / right_total * 100) if right_total > 0 else 0
    both_left_pct = (both_count / left_total * 100) if left_total > 0 else 0
    both_right_pct = (both_count / right_total * 100) if right_total > 0 else 0

    print(f"\n{'='*80}")
    print(f"COMMONALITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Left tree:  {left_hash}")
    print(f"Right tree: {right_hash}")
    print(f"\n{'='*80}")
    print(f"NODE DISTRIBUTION")
    print(f"{'='*80}")
    print(f"\nLeft tree total:  {left_total:,} nodes")
    print(f"  - Only in left: {left_only_count:,} nodes ({left_only_pct:.1f}%)")
    print(f"  - Shared:       {both_count:,} nodes ({both_left_pct:.1f}%)")

    print(f"\nRight tree total: {right_total:,} nodes")
    print(f"  - Only in right: {right_only_count:,} nodes ({right_only_pct:.1f}%)")
    print(f"  - Shared:        {both_count:,} nodes ({both_right_pct:.1f}%)")

    print(f"\n{'='*80}")
    print(f"VENN DIAGRAM SUMMARY")
    print(f"{'='*80}")
    print(f"Left only:  {left_only_count:,} nodes")
    print(f"Both:       {both_count:,} nodes")
    print(f"Right only: {right_only_count:,} nodes")
    print(f"{'='*80}")
