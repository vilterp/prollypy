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

"""Statistics tracking for ProllyTree node creation and sizes."""

from collections import defaultdict
from typing import Dict, List, Tuple


class Histogram:
    """Tracks size distribution for a set of values."""

    def __init__(self, name: str = ""):
        self.name = name
        self.counts: Dict[int, int] = defaultdict(int)
        self.total_count = 0
        self.total_size = 0

    def record(self, size: int):
        """Record a new value in the histogram."""
        self.counts[size] += 1
        self.total_count += 1
        self.total_size += size

    def get_average(self) -> float:
        """Return average size."""
        if self.total_count == 0:
            return 0.0
        return self.total_size / self.total_count

    def format_buckets(self, bucket_count: int = 10) -> List[Tuple[str, int]]:
        """Format histogram into buckets for display.

        Args:
            bucket_count: Number of buckets to create

        Returns:
            List of (range_label, count) tuples
        """
        if not self.counts:
            return []

        sizes = list(self.counts.keys())
        min_size = min(sizes)
        max_size = max(sizes)

        # Create buckets
        bucket_width = max(1, (max_size - min_size + 1) // bucket_count)
        buckets = defaultdict(int)

        for size, count in self.counts.items():
            bucket_idx = (size - min_size) // bucket_width
            bucket_start = min_size + bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width - 1
            bucket_key = f"{bucket_start}-{bucket_end}"
            buckets[bucket_key] += count

        # Sort by bucket start value
        sorted_buckets = sorted(buckets.items(), key=lambda x: int(x[0].split('-')[0]))
        return sorted_buckets

    def print_distribution(self, bucket_count: int = 10):
        """Print distribution of values."""
        if not self.counts:
            print(f"  No {self.name} recorded")
            return

        print(f"  {self.name} size distribution ({self.total_count:,} total):")
        buckets = self.format_buckets(bucket_count)

        for range_label, count in buckets:
            pct = (count * 100.0) / self.total_count
            bar = '#' * int(pct / 2)  # Scale bar to ~50 chars max
            print(f"    {range_label:>15} bytes: {count:>6,} ({pct:>5.1f}%) {bar}")


class Stats:
    """Tracks node creation statistics including counts and size distributions."""

    def __init__(self):
        # Histograms for leaf and internal nodes
        self.leaf_histogram = Histogram(name="Leaf node")
        self.internal_histogram = Histogram(name="Internal node")

    def record_new_leaf(self, size: int):
        """Record creation of a new leaf node with given serialized size in bytes."""
        self.leaf_histogram.record(size)

    def record_new_internal(self, size: int):
        """Record creation of a new internal node with given serialized size in bytes."""
        self.internal_histogram.record(size)

    def get_average_leaf_size(self) -> float:
        """Return average leaf size in bytes."""
        return self.leaf_histogram.get_average()

    def get_average_internal_size(self) -> float:
        """Return average internal node size in bytes."""
        return self.internal_histogram.get_average()

    def get_creation_stats(self) -> Dict[str, int]:
        """Return cumulative creation counts."""
        return {
            'total_leaves_created': self.leaf_histogram.total_count,
            'total_internals_created': self.internal_histogram.total_count
        }

    def get_size_stats(self) -> Dict[str, float]:
        """Return average size statistics."""
        return {
            'avg_leaf_size': self.get_average_leaf_size(),
            'avg_internal_size': self.get_average_internal_size()
        }

    def print_leaf_distribution(self, bucket_count: int = 10):
        """Print distribution of leaf sizes."""
        self.leaf_histogram.print_distribution(bucket_count)

    def print_internal_distribution(self, bucket_count: int = 10):
        """Print distribution of internal node sizes."""
        self.internal_histogram.print_distribution(bucket_count)

    def print_distributions(self, bucket_count: int = 10):
        """Print both leaf and internal node size distributions."""
        self.print_leaf_distribution(bucket_count)
        print()
        self.print_internal_distribution(bucket_count)
