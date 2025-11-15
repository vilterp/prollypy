# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Node class for ProllyTree.

A Node represents either a leaf or internal node in the tree structure.
"""

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Store


class Node:
    """
    Tree node - can be leaf or internal.

    ## Separator Semantics for Internal Nodes

    For internal nodes with n children, there are n-1 separator keys.
    Each separator key defines the boundary between adjacent children.

    **Key Invariant**: For an internal node:
    - `keys[i]` is the FIRST key in child `values[i+1]`
    - Child `values[i]` contains all keys k where:
      - If i == 0: k < keys[0] (all keys less than first separator)
      - If 0 < i < len(keys): keys[i-1] <= k < keys[i] (range between separators)
      - If i == len(keys): k >= keys[-1] (all keys >= last separator)

    **Example**: Internal node with keys=['d', 'h', 'm'] and 4 children:
    ```
    values[0]: all keys k where k < 'd'
    values[1]: all keys k where 'd' <= k < 'h'
    values[2]: all keys k where 'h' <= k < 'm'
    values[3]: all keys k where k >= 'm'
    ```

    This means:
    - `keys[0] = 'd'` is the first key in `values[1]`
    - `keys[1] = 'h'` is the first key in `values[2]`
    - `keys[2] = 'm'` is the first key in `values[3]`

    ## Seeking with Separators

    To find which child should contain a target key:
    ```python
    child_idx = 0
    for i, separator in enumerate(node.keys):
        if target >= separator:
            child_idx = i + 1
        else:
            break
    # Descend into values[child_idx]
    ```

    ## Leaf Nodes

    For leaf nodes:
    - `keys` contains the actual data keys
    - `values` contains the corresponding data values
    - len(keys) == len(values)
    """

    def __init__(self, is_leaf: bool = True) -> None:
        self.is_leaf: bool = is_leaf
        self.keys: "list[str]" = []      # Separator keys (for internal) or actual keys (for leaves)
        self.values: "list[str]" = []    # Child pointers (for internal) or actual values (for leaves)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf({list(zip(self.keys, self.values))})"
        else:
            return f"Internal(keys={self.keys}, children={len(self.values)})"

    def validate(self, store: Optional['Store'] = None, context: str = "") -> None:
        """
        Validate this node and its entire subtree.

        Recursively traverses all keys in sorted order and checks for duplicates/ordering.
        Also validates separator invariants for internal nodes.

        Args:
            store: Store to retrieve child nodes (required for internal nodes)
            context: Optional context string for error messages

        Raises:
            ValueError: If the node or its subtree is invalid
        """
        context_str = f" ({context})" if context else ""

        # Basic structural checks
        if self.is_leaf:
            if len(self.keys) != len(self.values):
                raise ValueError(f"Leaf node has {len(self.keys)} keys but {len(self.values)} values{context_str}")
        else:
            if len(self.values) != len(self.keys) + 1:
                raise ValueError(f"Internal node has {len(self.keys)} keys but {len(self.values)} children (should be keys+1){context_str}")

            # Validate separator invariants for internal nodes
            if store:
                self._validate_separators(store, context_str)

        # Collect all keys from this subtree in order
        all_keys = []
        self._collect_keys(all_keys, store)

        # Check that all keys are in sorted order with no duplicates
        prev_key = None
        for i, key in enumerate(all_keys):
            if prev_key is not None:
                if key == prev_key:
                    raise ValueError(f"Duplicate key at position {i}: {key}{context_str}")
                elif key < prev_key:
                    # Print debug info about the node
                    error_msg = [f"Keys out of order at position {i}: {prev_key} > {key}{context_str}"]
                    error_msg.append(f"\nNode structure:")
                    error_msg.append(f"  is_leaf: {self.is_leaf}")
                    error_msg.append(f"  num_keys: {len(self.keys)}")
                    error_msg.append(f"  num_values: {len(self.values)}")
                    if not self.is_leaf:
                        error_msg.append(f"  separator_keys: {self.keys}")
                        # Print first key of each child
                        if store:
                            error_msg.append(f"\n  Children first keys:")
                            for j, child_hash in enumerate(self.values):
                                child = store.get_node(child_hash)
                                if child and len(child.keys) > 0:
                                    first_key = child.keys[0] if child.is_leaf else child.keys[0] if child.keys else "<no keys>"
                                    error_msg.append(f"    child[{j}]: {first_key}")
                    else:
                        error_msg.append(f"  keys: {self.keys[:10]}..." if len(self.keys) > 10 else f"  keys: {self.keys}")
                    raise ValueError("\n".join(error_msg))
            prev_key = key

    def _validate_separators(self, store: 'Store', context_str: str) -> None:
        """
        Validate separator invariants for internal nodes.

        Checks that each separator key equals the first key in its corresponding child.
        """
        for i, separator in enumerate(self.keys):
            # separator should be the first key in child i+1
            child_idx = i + 1
            if child_idx >= len(self.values):
                raise ValueError(f"Separator {i} points to non-existent child {child_idx}{context_str}")

            child_hash = self.values[child_idx]
            child = store.get_node(child_hash)
            if child is None:
                raise ValueError(f"Child {child_idx} not found in store{context_str}")

            # Get first key from child
            first_key = self._get_first_key(child, store)
            if first_key is None:
                raise ValueError(f"Child {child_idx} has no keys{context_str}")

            if separator != first_key:
                raise ValueError(
                    f"Separator invariant violated at index {i}{context_str}\n"
                    f"  Expected separator: {first_key}\n"
                    f"  Actual separator: {separator}\n"
                    f"  Child index: {child_idx}"
                )

    def _get_first_key(self, node: 'Node', store: 'Store') -> Optional[str]:
        """Get the first key in a node's subtree."""
        if node.is_leaf:
            return node.keys[0] if len(node.keys) > 0 else None
        else:
            # Descend to leftmost child
            if len(node.values) == 0:
                return None
            child_hash = node.values[0]
            child = store.get_node(child_hash)
            if child is None:
                return None
            return self._get_first_key(child, store)

    def _collect_keys(self, result: "list[str]", store: Optional['Store']) -> None:
        """Recursively collect all keys from this node's subtree in traversal order."""
        if self.is_leaf:
            # For leaf nodes, just add all keys
            result.extend(self.keys)
        else:
            # For internal nodes, traverse children in order
            for i, child_hash in enumerate(self.values):
                # Get child from store
                if store is None:
                    raise ValueError("Cannot validate internal node without store")
                child = store.get_node(child_hash)
                if child is None:
                    raise ValueError(f"Child node {child_hash} not found in store")
                # Recursively collect keys from child
                child._collect_keys(result, store)
