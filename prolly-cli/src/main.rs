mod sqlite_commit_store;

use clap::{Parser, Subcommand};
use prolly_core::{garbage_collect, BlockStore, CachedFSBlockStore, DB, Repo};
use rusqlite::Connection;
use sqlite_commit_store::SqliteCommitGraphStore;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "prolly")]
#[command(about = "ProllyTree database and version control", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Import a SQLite database
    Import {
        /// Path to SQLite database file
        sqlite_path: PathBuf,

        /// Path to prolly repository (will be created)
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Batch size for inserts (0 = single batch)
        #[arg(short, long, default_value = "10000")]
        batch_size: usize,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Garbage collect unreachable nodes
    Gc {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Dry run - don't actually remove nodes, just report statistics
        #[arg(short, long)]
        dry_run: bool,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// List or create branches
    Branch {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Branch name to create (if not provided, lists all branches)
        name: Option<String>,

        /// Create branch from this commit (default: HEAD)
        #[arg(short, long)]
        from: Option<String>,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Switch to a different branch
    Checkout {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Branch name to checkout
        branch: String,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Show commit history
    Log {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Start from this ref (default: HEAD)
        #[arg(short, long)]
        start: Option<String>,

        /// Maximum number of commits to show
        #[arg(short, long)]
        max_count: Option<usize>,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },
}

fn import_sqlite(
    sqlite_path: PathBuf,
    repo_path: PathBuf,
    batch_size: usize,
    cache_size: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    println!("Importing SQLite database: {}", sqlite_path.display());
    println!("Target repository: {}", repo_path.display());
    println!("Batch size: {}", batch_size);
    println!("Cache size: {}", cache_size);
    println!();

    // Create repository directory
    std::fs::create_dir_all(&repo_path)?;

    // Open SQLite database
    let sqlite_conn = Connection::open(&sqlite_path)?;

    // Get list of tables
    let mut stmt = sqlite_conn.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    )?;
    let table_names: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;

    println!("Found {} tables", table_names.len());
    println!();

    // Create block store
    let store_path = repo_path.join("blocks");
    let store = Arc::new(CachedFSBlockStore::new(&store_path, cache_size)?);

    // Create database
    let mut db = DB::new(store.clone(), 0.01, 42);

    let total_start = Instant::now();
    let mut total_rows = 0;

    // Import each table
    for table_name in &table_names {
        println!("=== Importing table: {} ===", table_name);

        // Get table schema
        let mut stmt = sqlite_conn.prepare(&format!("PRAGMA table_info({})", table_name))?;
        let table_info: Vec<_> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(1)?, // name
                    row.get::<_, String>(2)?, // type
                    row.get::<_, i32>(5)?,    // pk
                ))
            })?
            .collect::<Result<_, rusqlite::Error>>()?;

        let columns: Vec<String> = table_info.iter().map(|(name, _, _)| name.clone()).collect();
        let types: Vec<String> = table_info
            .iter()
            .map(|(_, col_type, _)| col_type.clone())
            .collect();
        let primary_key: Vec<String> = table_info
            .iter()
            .filter(|(_, _, pk)| *pk > 0)
            .map(|(name, _, _)| name.clone())
            .collect();

        println!("Columns: {:?}", columns);
        println!("Primary key: {:?}", primary_key);

        // Create table in DB
        db.create_table(table_name.clone(), columns.clone(), types, primary_key);

        // Count rows
        let row_count: i64 = sqlite_conn
            .query_row(&format!("SELECT COUNT(*) FROM {}", table_name), [], |row| {
                row.get(0)
            })?;
        println!("Row count: {}", row_count);

        if row_count == 0 {
            println!("Skipping empty table");
            println!();
            continue;
        }

        // Import data in batches
        let column_list = columns.join(", ");
        let query = format!("SELECT {} FROM {}", column_list, table_name);

        let mut stmt = sqlite_conn.prepare(&query)?;

        // Fetch all rows (rusqlite doesn't support streaming easily)
        let rows: Vec<Vec<serde_json::Value>> = stmt
            .query_map([], |row| {
                let mut values = Vec::new();
                for i in 0..columns.len() {
                    let value: serde_json::Value = match row.get::<_, rusqlite::types::Value>(i)? {
                        rusqlite::types::Value::Null => serde_json::Value::Null,
                        rusqlite::types::Value::Integer(i) => serde_json::Value::from(i),
                        rusqlite::types::Value::Real(f) => serde_json::Value::from(f),
                        rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
                        rusqlite::types::Value::Blob(b) => {
                            serde_json::Value::String(String::from_utf8_lossy(&b).to_string())
                        }
                    };
                    values.push(value);
                }
                Ok(values)
            })?
            .collect::<Result<_, rusqlite::Error>>()?;

        let table_start = Instant::now();

        if batch_size > 0 && rows.len() > batch_size {
            // Insert in batches
            for (i, chunk) in rows.chunks(batch_size).enumerate() {
                let batch_start = Instant::now();
                db.insert_rows(table_name, chunk.to_vec(), verbose);
                let batch_elapsed = batch_start.elapsed();
                let rows_per_sec = chunk.len() as f64 / batch_elapsed.as_secs_f64();

                if verbose {
                    println!(
                        "Batch {}: {} rows in {:.2}s ({:.0} rows/sec)",
                        i + 1,
                        chunk.len(),
                        batch_elapsed.as_secs_f64(),
                        rows_per_sec
                    );
                }

                total_rows += chunk.len();
            }
        } else {
            // Single batch
            db.insert_rows(table_name, rows.clone(), verbose);
            total_rows += rows.len();
        }

        let table_elapsed = table_start.elapsed();
        let rows_per_sec = rows.len() as f64 / table_elapsed.as_secs_f64();

        println!(
            "Imported {} rows in {:.2}s ({:.0} rows/sec)",
            rows.len(),
            table_elapsed.as_secs_f64(),
            rows_per_sec
        );
        println!();
    }

    let total_elapsed = total_start.elapsed();
    let total_rows_per_sec = total_rows as f64 / total_elapsed.as_secs_f64();

    println!("=== Import Complete ===");
    println!("Total rows: {}", total_rows);
    println!("Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("Average: {:.0} rows/sec", total_rows_per_sec);
    println!();
    println!("Root hash: {}", hex::encode(db.get_root_hash()));
    println!("Total nodes: {}", store.count_nodes());

    Ok(())
}

fn gc_repo(repo_path: PathBuf, dry_run: bool, cache_size: usize) -> anyhow::Result<()> {
    println!("Garbage collecting repository: {}", repo_path.display());
    println!("Mode: {}", if dry_run { "dry-run" } else { "live" });

    // Open the repository
    let store_path = repo_path.join("blocks");
    let store = Arc::new(CachedFSBlockStore::new(&store_path, cache_size)?);
    let commit_store_path = repo_path.join("commits.db");
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path)?);
    let repo = Repo::new(store.clone(), commit_store, "prolly-cli".to_string());

    println!();
    println!("Finding reachable tree roots...");
    let tree_roots = repo.get_reachable_tree_roots();
    println!("Found {} reachable tree roots", tree_roots.len());

    println!();
    println!("Running garbage collection...");
    let start = Instant::now();
    let stats = garbage_collect(store.as_ref(), &tree_roots, dry_run);
    let elapsed = start.elapsed();

    println!();
    println!("=== Garbage Collection Results ===");
    println!("Total nodes: {}", stats.total_nodes);
    println!("Reachable nodes: {}", stats.reachable_nodes);
    println!("Garbage nodes: {}", stats.garbage_nodes);
    println!(
        "Reachable: {:.1}%",
        stats.reachable_percent()
    );
    println!(
        "Garbage: {:.1}%",
        stats.garbage_percent()
    );
    println!("Time: {:.2}s", elapsed.as_secs_f64());

    if dry_run {
        println!();
        println!("Dry run complete - no nodes were removed");
        println!("Run without --dry-run to actually remove garbage nodes");
    } else {
        println!();
        println!("Removed {} garbage nodes", stats.garbage_nodes);
    }

    Ok(())
}

fn branch_cmd(
    repo_path: PathBuf,
    name: Option<String>,
    from: Option<String>,
    cache_size: usize,
) -> anyhow::Result<()> {
    let store_path = repo_path.join("blocks");
    let store = Arc::new(CachedFSBlockStore::new(&store_path, cache_size)?);
    let commit_store_path = repo_path.join("commits.db");
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path)?);
    let repo = Repo::new(store, commit_store, "prolly-cli".to_string());

    if let Some(branch_name) = name {
        // Create a new branch
        let from_commit = if let Some(from_ref) = from {
            Some(repo.resolve_ref(&from_ref).ok_or_else(|| {
                anyhow::anyhow!("Could not resolve ref: {}", from_ref)
            })?)
        } else {
            None
        };

        repo.create_branch(&branch_name, from_commit.as_ref())
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        println!("Created branch: {}", branch_name);
    } else {
        // List all branches
        let (head_commit, head_ref) = repo.get_head();
        let branches = repo.list_branches();

        if branches.is_empty() {
            println!("No branches found");
        } else {
            for (name, hash) in branches {
                let marker = if name == head_ref { "* " } else { "  " };
                println!("{}{} ({})", marker, name, hex::encode(&hash[..8]));
            }

            if let Some(commit) = head_commit {
                println!();
                println!("HEAD -> {}", head_ref);
                println!("Commit: {}", commit.message);
            }
        }
    }

    Ok(())
}

fn checkout_cmd(repo_path: PathBuf, branch: String, cache_size: usize) -> anyhow::Result<()> {
    let store_path = repo_path.join("blocks");
    let store = Arc::new(CachedFSBlockStore::new(&store_path, cache_size)?);
    let commit_store_path = repo_path.join("commits.db");
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path)?);
    let repo = Repo::new(store, commit_store, "prolly-cli".to_string());

    repo.checkout(&branch)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Switched to branch: {}", branch);

    // Show commit info
    let (commit, _) = repo.get_head();
    if let Some(commit) = commit {
        println!("Commit: {}", hex::encode(&commit.compute_hash()[..8]));
        println!("Message: {}", commit.message);
    }

    Ok(())
}

fn log_cmd(
    repo_path: PathBuf,
    start: Option<String>,
    max_count: Option<usize>,
    cache_size: usize,
) -> anyhow::Result<()> {
    let store_path = repo_path.join("blocks");
    let store = Arc::new(CachedFSBlockStore::new(&store_path, cache_size)?);
    let commit_store_path = repo_path.join("commits.db");
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path)?);
    let repo = Repo::new(store, commit_store, "prolly-cli".to_string());

    let commits = repo.log(start.as_deref(), max_count);

    if commits.is_empty() {
        println!("No commits found");
        return Ok(());
    }

    for (hash, commit) in commits {
        println!("commit {}", hex::encode(&hash));
        println!("Author: {}", commit.author);
        println!(
            "Date:   {}",
            chrono::DateTime::from_timestamp(commit.timestamp as i64, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| commit.timestamp.to_string())
        );
        println!();
        println!("    {}", commit.message);
        println!();
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Import {
            sqlite_path,
            repo,
            batch_size,
            cache_size,
            verbose,
        } => import_sqlite(sqlite_path, repo, batch_size, cache_size, verbose)?,
        Commands::Gc {
            repo,
            dry_run,
            cache_size,
        } => gc_repo(repo, dry_run, cache_size)?,
        Commands::Branch {
            repo,
            name,
            from,
            cache_size,
        } => branch_cmd(repo, name, from, cache_size)?,
        Commands::Checkout {
            repo,
            branch,
            cache_size,
        } => checkout_cmd(repo, branch, cache_size)?,
        Commands::Log {
            repo,
            start,
            max_count,
            cache_size,
        } => log_cmd(repo, start, max_count, cache_size)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prolly_core::ProllyTree;
    use tempfile::TempDir;

    fn create_test_repo(repo_path: &PathBuf) -> Repo {
        let store_path = repo_path.join("blocks");
        let store = Arc::new(CachedFSBlockStore::new(&store_path, 100).unwrap());
        let commit_store_path = repo_path.join("commits.db");
        let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path).unwrap());

        Repo::init_empty(store, commit_store, "test@example.com".to_string())
    }

    #[test]
    fn test_import_sqlite_basic() {
        let temp_dir = TempDir::new().unwrap();
        let sqlite_path = temp_dir.path().join("test.db");
        let repo_path = temp_dir.path().join("repo");

        // Create a test SQLite database
        let conn = Connection::open(&sqlite_path).unwrap();
        conn.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL
            )",
            [],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@example.com')",
            [],
        )
        .unwrap();
        conn.close().unwrap();

        // Call the import function
        let result = import_sqlite(sqlite_path, repo_path.clone(), 0, 100, false);
        assert!(result.is_ok());

        // Verify the repo was created
        assert!(repo_path.join("blocks").exists());
    }

    #[test]
    fn test_branch_create_and_list() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create a test commit
        let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree.insert_batch(vec![(b"key1".to_vec(), b"value1".to_vec())], false);
        let tree_root = tree.get_root_hash();

        repo.commit(&tree_root, "Initial commit", None, None, None);

        // Create a new branch
        let head_commit = repo.resolve_ref("main").unwrap();
        repo.create_branch("feature", Some(&head_commit)).unwrap();

        // List branches
        let branches = repo.list_branches();
        assert_eq!(branches.len(), 2);
        assert!(branches.iter().any(|(name, _)| name == "main"));
        assert!(branches.iter().any(|(name, _)| name == "feature"));

        // Both should point to the same commit
        let main_hash = branches.iter().find(|(name, _)| name == "main").map(|(_, hash)| hash);
        let feature_hash = branches.iter().find(|(name, _)| name == "feature").map(|(_, hash)| hash);
        assert_eq!(main_hash, feature_hash);
    }

    #[test]
    fn test_checkout_branch() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create initial commit on main
        let mut tree1 = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree1.insert_batch(vec![(b"key1".to_vec(), b"value1".to_vec())], false);
        let tree_root1 = tree1.get_root_hash();
        repo.commit(&tree_root1, "First commit", None, None, None);

        // Create a branch and switch to it
        let main_commit = repo.resolve_ref("main").unwrap();
        repo.create_branch("develop", Some(&main_commit)).unwrap();
        repo.checkout("develop").unwrap();

        // Verify we're on develop
        let (_, head_ref) = repo.get_head();
        assert_eq!(head_ref, "develop");

        // Create a commit on develop
        let mut tree2 = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree2.insert_batch(
            vec![
                (b"key1".to_vec(), b"value1".to_vec()),
                (b"key2".to_vec(), b"value2".to_vec()),
            ],
            false,
        );
        let tree_root2 = tree2.get_root_hash();
        repo.commit(&tree_root2, "Second commit on develop", None, None, None);

        // Switch back to main
        repo.checkout("main").unwrap();
        let (_, head_ref) = repo.get_head();
        assert_eq!(head_ref, "main");

        // Main should still point to first commit
        let (main_commit, _) = repo.get_head();
        assert_eq!(main_commit.unwrap().message, "First commit");
    }

    #[test]
    fn test_log_commits() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create multiple commits
        for i in 1..=3 {
            let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
            tree.insert_batch(
                vec![(format!("key{}", i).into_bytes(), b"value".to_vec())],
                false,
            );
            let tree_root = tree.get_root_hash();
            repo.commit(&tree_root, &format!("Commit {}", i), None, None, None);
        }

        // Get log from HEAD
        let commits = repo.log(None, None);
        assert_eq!(commits.len(), 4); // 3 commits + initial commit

        // Commits should be in reverse chronological order
        assert_eq!(commits[0].1.message, "Commit 3");
        assert_eq!(commits[1].1.message, "Commit 2");
        assert_eq!(commits[2].1.message, "Commit 1");
        assert_eq!(commits[3].1.message, "Initial commit");

        // Test with max_count
        let commits = repo.log(None, Some(2));
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].1.message, "Commit 3");
        assert_eq!(commits[1].1.message, "Commit 2");
    }

    #[test]
    fn test_gc_no_garbage() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create a commit
        let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree.insert_batch(
            vec![
                (b"a".to_vec(), b"val_a".to_vec()),
                (b"b".to_vec(), b"val_b".to_vec()),
            ],
            false,
        );
        let tree_root = tree.get_root_hash();
        repo.commit(&tree_root, "Test commit", None, None, None);

        // Run GC - should find no garbage since all nodes are reachable
        let tree_roots = repo.get_reachable_tree_roots();
        let stats = garbage_collect(repo.block_store.as_ref(), &tree_roots, true);

        assert_eq!(stats.garbage_nodes, 0);
        assert!(stats.reachable_nodes > 0);
    }

    #[test]
    fn test_gc_with_garbage() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create first commit
        let mut tree1 = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree1.insert_batch(vec![(b"key1".to_vec(), b"value1".to_vec())], false);
        let tree_root1 = tree1.get_root_hash();
        repo.commit(&tree_root1, "First commit", None, None, None);

        // Create an orphaned tree (not committed)
        let mut orphan_tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        orphan_tree.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);
        let _orphan_root = orphan_tree.get_root_hash();

        let nodes_before = repo.block_store.count_nodes();

        // Run GC - should find the orphaned nodes
        let tree_roots = repo.get_reachable_tree_roots();
        let stats = garbage_collect(repo.block_store.as_ref(), &tree_roots, false);

        assert!(stats.garbage_nodes > 0, "Should have found garbage nodes");

        let nodes_after = repo.block_store.count_nodes();
        assert!(nodes_after < nodes_before, "Should have removed nodes");
        assert_eq!(nodes_after, stats.reachable_nodes);
    }

    #[test]
    fn test_gc_dry_run() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create commit and orphan
        let mut tree1 = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree1.insert_batch(vec![(b"key1".to_vec(), b"value1".to_vec())], false);
        let tree_root1 = tree1.get_root_hash();
        repo.commit(&tree_root1, "First commit", None, None, None);

        let mut orphan_tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        orphan_tree.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);

        let nodes_before = repo.block_store.count_nodes();

        // Run GC in dry-run mode
        let tree_roots = repo.get_reachable_tree_roots();
        let stats = garbage_collect(repo.block_store.as_ref(), &tree_roots, true);

        let nodes_after = repo.block_store.count_nodes();

        // Dry run should not remove anything
        assert_eq!(nodes_before, nodes_after);
        assert!(stats.garbage_nodes > 0, "Should have reported garbage nodes");
    }

    #[test]
    fn test_resolve_ref_head_tilde() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create multiple commits
        let mut commits = Vec::new();
        for i in 1..=3 {
            let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
            tree.insert_batch(
                vec![(format!("key{}", i).into_bytes(), b"value".to_vec())],
                false,
            );
            let tree_root = tree.get_root_hash();
            let commit = repo.commit(&tree_root, &format!("Commit {}", i), None, None, None);
            let commit_hash = commit.compute_hash();
            commits.push(commit_hash);
        }

        // Test HEAD resolves to latest commit
        let head = repo.resolve_ref("HEAD").unwrap();
        assert_eq!(head, commits[2]); // commits[2] is "Commit 3"

        // Test HEAD~1 resolves to parent
        let head_1 = repo.resolve_ref("HEAD~1").unwrap();
        assert_eq!(head_1, commits[1]);

        // Test HEAD~2
        let head_2 = repo.resolve_ref("HEAD~2").unwrap();
        assert_eq!(head_2, commits[0]);
    }

    #[test]
    fn test_branch_cmd_creates_branch() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create a commit
        let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree.insert_batch(vec![(b"test".to_vec(), b"data".to_vec())], false);
        let tree_root = tree.get_root_hash();
        repo.commit(&tree_root, "Test", None, None, None);

        // Use the CLI function to create a branch
        let result = branch_cmd(repo_path.clone(), Some("feature".to_string()), None, 100);
        assert!(result.is_ok());

        // Verify branch was created
        let branches = repo.list_branches();
        assert!(branches.iter().any(|(name, _)| name == "feature"));
    }

    #[test]
    fn test_checkout_cmd_switches_branch() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create commits and branch
        let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree.insert_batch(vec![(b"test".to_vec(), b"data".to_vec())], false);
        let tree_root = tree.get_root_hash();
        repo.commit(&tree_root, "Test", None, None, None);

        let main_commit = repo.resolve_ref("main").unwrap();
        repo.create_branch("develop", Some(&main_commit)).unwrap();

        // Use the CLI function to checkout
        let result = checkout_cmd(repo_path.clone(), "develop".to_string(), 100);
        assert!(result.is_ok());

        // Verify we switched
        let (_, head_ref) = repo.get_head();
        assert_eq!(head_ref, "develop");
    }

    #[test]
    fn test_gc_repo_dry_run() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Create commit and orphan
        let mut tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        tree.insert_batch(vec![(b"test".to_vec(), b"data".to_vec())], false);
        let tree_root = tree.get_root_hash();
        repo.commit(&tree_root, "Test", None, None, None);

        let mut orphan = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        orphan.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);

        let nodes_before = repo.block_store.count_nodes();

        // Run GC via CLI function
        let result = gc_repo(repo_path.clone(), true, 100);
        assert!(result.is_ok());

        // Verify dry run didn't remove nodes
        let nodes_after = repo.block_store.count_nodes();
        assert_eq!(nodes_before, nodes_after);
    }
}
