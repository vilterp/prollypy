use prolly_core::ProllyTree;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("ProllyTree Performance Benchmark\n");
    println!("=================================\n");

    // Benchmark 1: Insert 10,000 items
    {
        let mut tree = ProllyTree::default();
        let mut mutations = Vec::new();

        for i in 0..10_000 {
            let key = format!("key{:06}", i).into_bytes();
            let value = format!("value{}", i).into_bytes();
            mutations.push((key, value));
        }

        let start = Instant::now();
        tree.insert_batch(mutations, false);
        let elapsed = start.elapsed();

        println!("Insert 10,000 items:");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Rate: {:.0} inserts/sec", 10_000.0 / elapsed.as_secs_f64());
        println!("  Final count: {}", tree.count());
        let stats = tree.get_stats();
        println!("  Nodes created: {}", stats.nodes_created);
        println!("  Leaves created: {}", stats.leaves_created);
        println!("  Internals created: {}", stats.internals_created);
        println!();
    }

    // Benchmark 2: Insert 100,000 items
    {
        let mut tree = ProllyTree::default();
        let mut mutations = Vec::new();

        for i in 0..100_000 {
            let key = format!("key{:06}", i).into_bytes();
            let value = format!("value{}", i).into_bytes();
            mutations.push((key, value));
        }

        let start = Instant::now();
        tree.insert_batch(mutations, false);
        let elapsed = start.elapsed();

        println!("Insert 100,000 items:");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Rate: {:.0} inserts/sec", 100_000.0 / elapsed.as_secs_f64());
        println!("  Final count: {}", tree.count());
        let stats = tree.get_stats();
        println!("  Nodes created: {}", stats.nodes_created);
        println!("  Leaves created: {}", stats.leaves_created);
        println!("  Internals created: {}", stats.internals_created);
        println!();
    }

    // Benchmark 3: Incremental updates
    {
        let mut tree = ProllyTree::default();

        // Initial insert
        let mut mutations = Vec::new();
        for i in 0..50_000 {
            let key = format!("key{:06}", i).into_bytes();
            let value = format!("value{}", i).into_bytes();
            mutations.push((key, value));
        }
        tree.insert_batch(mutations, false);

        // Update half of the keys
        let mut updates = Vec::new();
        for i in 0..25_000 {
            let key = format!("key{:06}", i).into_bytes();
            let value = format!("updated{}", i).into_bytes();
            updates.push((key, value));
        }

        let start = Instant::now();
        tree.insert_batch(updates, false);
        let elapsed = start.elapsed();

        println!("Update 25,000 items (out of 50,000):");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Rate: {:.0} updates/sec", 25_000.0 / elapsed.as_secs_f64());
        let stats = tree.get_stats();
        println!("  Nodes created: {}", stats.nodes_created);
        println!("  Nodes reused: {}", stats.nodes_reused);
        println!();
    }

    println!("Benchmark complete!");
}
