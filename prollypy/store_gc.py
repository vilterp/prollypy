"""
Garbage collection for ProllyTree.

Identifies unreachable nodes in the store by traversing from root hashes
and comparing with all stored nodes.
"""

from typing import Set, Iterator
from dataclasses import dataclass
from .store import Store


@dataclass
class GCStats:
    """Statistics from a garbage collection operation."""
    total_nodes: int = 0
    reachable_nodes: int = 0
    garbage_nodes: int = 0

    @property
    def reachable_percent(self) -> float:
        """Percentage of nodes that are reachable."""
        if self.total_nodes == 0:
            return 0.0
        return (self.reachable_nodes / self.total_nodes) * 100

    @property
    def garbage_percent(self) -> float:
        """Percentage of nodes that are garbage."""
        if self.total_nodes == 0:
            return 0.0
        return (self.garbage_nodes / self.total_nodes) * 100

    def __repr__(self):
        return (
            f"GCStats(total={self.total_nodes}, "
            f"reachable={self.reachable_nodes} ({self.reachable_percent:.1f}%), "
            f"garbage={self.garbage_nodes} ({self.garbage_percent:.1f}%))"
        )


def find_reachable_nodes(store: Store, root_hashes: Set[str]) -> Set[str]:
    """
    Find all nodes reachable from a set of root hashes.

    Performs a depth-first traversal from each root, visiting all
    descendant nodes and collecting their hashes.

    Args:
        store: Storage backend containing nodes
        root_hashes: Set of root hashes to start traversal from

    Returns:
        Set of all reachable node hashes (including roots)
    """
    reachable = set()
    to_visit = list(root_hashes)

    while to_visit:
        node_hash = to_visit.pop()

        # Skip if already visited
        if node_hash in reachable:
            continue

        # Get node and traverse children
        node = store.get_node(node_hash)
        if node is None:
            # Node not found in store - skip it (don't mark as reachable)
            continue

        # Mark as reachable (only if node exists)
        reachable.add(node_hash)

        if not node.is_leaf:
            # Internal node - add all children to visit
            for child_hash in node.values:
                if child_hash not in reachable:
                    to_visit.append(child_hash)

    return reachable


def iter_all_nodes(store: Store) -> Iterator[str]:
    """
    Iterate over all node hashes in the store.

    Args:
        store: Storage backend to query

    Yields:
        Node hashes present in the store
    """
    yield from store.list_nodes()


def find_garbage_nodes(store: Store, root_hashes: Set[str]) -> Set[str]:
    """
    Find all garbage (unreachable) nodes in the store.

    Args:
        store: Storage backend containing nodes
        root_hashes: Set of root hashes to keep (reachable roots)

    Returns:
        Set of garbage node hashes (unreachable from any root)
    """
    # Find all reachable nodes
    reachable = find_reachable_nodes(store, root_hashes)

    # Find all nodes in store
    all_nodes = set(iter_all_nodes(store))

    # Garbage = all nodes - reachable nodes
    garbage = all_nodes - reachable

    return garbage


def collect_garbage_stats(store: Store, root_hashes: Set[str]) -> GCStats:
    """
    Compute garbage collection statistics without removing anything.

    Args:
        store: Storage backend containing nodes
        root_hashes: Set of root hashes to keep

    Returns:
        GCStats with information about reachable and garbage nodes
    """
    # Find reachable and all nodes
    reachable = find_reachable_nodes(store, root_hashes)
    all_nodes = set(iter_all_nodes(store))

    # Compute statistics
    stats = GCStats(
        total_nodes=len(all_nodes),
        reachable_nodes=len(reachable),
        garbage_nodes=len(all_nodes) - len(reachable)
    )

    return stats


def remove_garbage(store: Store, garbage_hashes: Set[str]) -> int:
    """
    Remove garbage nodes from the store.

    Note: This is a destructive operation and should only be called
    after verifying that the nodes are truly garbage.

    Args:
        store: Storage backend to remove nodes from
        garbage_hashes: Set of node hashes to remove

    Returns:
        Number of nodes actually removed
    """
    removed_count = 0

    for node_hash in garbage_hashes:
        if store.delete_node(node_hash):
            removed_count += 1

    return removed_count


def garbage_collect(store: Store, root_hashes: Set[str], dry_run: bool = True) -> GCStats:
    """
    Perform garbage collection on a store.

    Args:
        store: Storage backend to garbage collect
        root_hashes: Set of root hashes to keep (everything else is garbage)
        dry_run: If True, only compute statistics without removing nodes

    Returns:
        GCStats with information about the garbage collection operation
    """
    # Compute statistics
    stats = collect_garbage_stats(store, root_hashes)

    # If not dry run, actually remove garbage
    if not dry_run:
        garbage = find_garbage_nodes(store, root_hashes)
        removed_count = remove_garbage(store, garbage)

        # Verify removal count matches expectations
        if removed_count != stats.garbage_nodes:
            raise RuntimeError(
                f"Expected to remove {stats.garbage_nodes} nodes, "
                f"but actually removed {removed_count}"
            )

    return stats
