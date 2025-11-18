//! Statistics tracking for ProllyTree node creation and sizes.

use std::collections::HashMap;

/// Tracks size distribution for a set of values.
#[derive(Debug, Clone)]
pub struct Histogram {
    pub name: String,
    pub counts: HashMap<usize, usize>,
    pub total_count: usize,
    pub total_size: usize,
}

impl Histogram {
    /// Create a new histogram with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Histogram {
            name: name.into(),
            counts: HashMap::new(),
            total_count: 0,
            total_size: 0,
        }
    }

    /// Record a new value in the histogram.
    pub fn record(&mut self, size: usize) {
        *self.counts.entry(size).or_insert(0) += 1;
        self.total_count += 1;
        self.total_size += size;
    }

    /// Return average size.
    pub fn get_average(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.total_size as f64 / self.total_count as f64
        }
    }

    /// Format histogram into buckets for display.
    ///
    /// # Arguments
    ///
    /// * `bucket_count` - Number of buckets to create
    ///
    /// # Returns
    ///
    /// List of (range_label, count) tuples
    pub fn format_buckets(&self, bucket_count: usize) -> Vec<(String, usize)> {
        if self.counts.is_empty() {
            return Vec::new();
        }

        let sizes: Vec<usize> = self.counts.keys().copied().collect();
        let min_size = *sizes.iter().min().unwrap();
        let max_size = *sizes.iter().max().unwrap();

        // Create buckets
        let bucket_width = std::cmp::max(1, (max_size - min_size + 1) / bucket_count);
        let mut buckets: HashMap<String, usize> = HashMap::new();

        for (&size, &count) in &self.counts {
            let bucket_idx = (size - min_size) / bucket_width;
            let bucket_start = min_size + bucket_idx * bucket_width;
            let bucket_end = bucket_start + bucket_width - 1;
            let bucket_key = format!("{}-{}", bucket_start, bucket_end);
            *buckets.entry(bucket_key).or_insert(0) += count;
        }

        // Sort by bucket start value
        let mut sorted_buckets: Vec<_> = buckets.into_iter().collect();
        sorted_buckets.sort_by_key(|(label, _)| {
            label
                .split('-')
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
        });

        sorted_buckets
    }

    /// Print distribution of values.
    pub fn print_distribution(&self, bucket_count: usize) {
        if self.counts.is_empty() {
            println!("  No {} recorded", self.name);
            return;
        }

        println!(
            "  {} size distribution ({} total):",
            self.name,
            format_with_commas(self.total_count)
        );
        let buckets = self.format_buckets(bucket_count);

        for (range_label, count) in buckets {
            let pct = (count as f64 * 100.0) / self.total_count as f64;
            let bar = "#".repeat((pct / 2.0) as usize); // Scale bar to ~50 chars max
            println!(
                "    {:>15} bytes: {:>6} ({:>5.1}%) {}",
                range_label,
                format_with_commas(count),
                pct,
                bar
            );
        }
    }
}

/// Tracks node creation statistics including counts and size distributions.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Histogram for leaf nodes
    pub leaf_histogram: Histogram,
    /// Histogram for internal nodes
    pub internal_histogram: Histogram,
}

impl Stats {
    /// Create a new Stats instance.
    pub fn new() -> Self {
        Stats {
            leaf_histogram: Histogram::new("Leaf node"),
            internal_histogram: Histogram::new("Internal node"),
        }
    }

    /// Record creation of a new leaf node with given serialized size in bytes.
    pub fn record_new_leaf(&mut self, size: usize) {
        self.leaf_histogram.record(size);
    }

    /// Record creation of a new internal node with given serialized size in bytes.
    pub fn record_new_internal(&mut self, size: usize) {
        self.internal_histogram.record(size);
    }

    /// Return average leaf size in bytes.
    pub fn get_average_leaf_size(&self) -> f64 {
        self.leaf_histogram.get_average()
    }

    /// Return average internal node size in bytes.
    pub fn get_average_internal_size(&self) -> f64 {
        self.internal_histogram.get_average()
    }

    /// Return cumulative creation counts.
    pub fn get_creation_stats(&self) -> CreationStats {
        CreationStats {
            total_leaves_created: self.leaf_histogram.total_count,
            total_internals_created: self.internal_histogram.total_count,
        }
    }

    /// Return average size statistics.
    pub fn get_size_stats(&self) -> SizeStats {
        SizeStats {
            avg_leaf_size: self.get_average_leaf_size(),
            avg_internal_size: self.get_average_internal_size(),
        }
    }

    /// Print distribution of leaf sizes.
    pub fn print_leaf_distribution(&self, bucket_count: usize) {
        self.leaf_histogram.print_distribution(bucket_count);
    }

    /// Print distribution of internal node sizes.
    pub fn print_internal_distribution(&self, bucket_count: usize) {
        self.internal_histogram.print_distribution(bucket_count);
    }

    /// Print both leaf and internal node size distributions.
    pub fn print_distributions(&self, bucket_count: usize) {
        self.print_leaf_distribution(bucket_count);
        println!();
        self.print_internal_distribution(bucket_count);
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

/// Creation statistics
#[derive(Debug, Clone)]
pub struct CreationStats {
    pub total_leaves_created: usize,
    pub total_internals_created: usize,
}

/// Size statistics
#[derive(Debug, Clone)]
pub struct SizeStats {
    pub avg_leaf_size: f64,
    pub avg_internal_size: f64,
}

/// Helper function to format numbers with commas
fn format_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for ch in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
        count += 1;
    }

    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_record() {
        let mut hist = Histogram::new("test");
        hist.record(100);
        hist.record(100);
        hist.record(200);

        assert_eq!(hist.total_count, 3);
        assert_eq!(hist.total_size, 400);
        assert_eq!(hist.counts.get(&100), Some(&2));
        assert_eq!(hist.counts.get(&200), Some(&1));
    }

    #[test]
    fn test_histogram_average() {
        let mut hist = Histogram::new("test");
        hist.record(100);
        hist.record(200);
        hist.record(300);

        assert_eq!(hist.get_average(), 200.0);
    }

    #[test]
    fn test_histogram_average_empty() {
        let hist = Histogram::new("test");
        assert_eq!(hist.get_average(), 0.0);
    }

    #[test]
    fn test_stats_record() {
        let mut stats = Stats::new();
        stats.record_new_leaf(100);
        stats.record_new_leaf(200);
        stats.record_new_internal(300);

        let creation = stats.get_creation_stats();
        assert_eq!(creation.total_leaves_created, 2);
        assert_eq!(creation.total_internals_created, 1);

        let sizes = stats.get_size_stats();
        assert_eq!(sizes.avg_leaf_size, 150.0);
        assert_eq!(sizes.avg_internal_size, 300.0);
    }

    #[test]
    fn test_format_buckets() {
        let mut hist = Histogram::new("test");
        for i in 0..10 {
            hist.record(i * 10);
        }

        let buckets = hist.format_buckets(5);
        assert!(!buckets.is_empty());
    }

    #[test]
    fn test_format_with_commas() {
        assert_eq!(format_with_commas(1000), "1,000");
        assert_eq!(format_with_commas(1000000), "1,000,000");
        assert_eq!(format_with_commas(123), "123");
    }
}
