mod sqlite_commit_store;
mod s3_store;

use clap::{Parser, Subcommand};
use prolly_core::{garbage_collect, BlockStore, CachedFSBlockStore, ProllyTree, TreeCursor, DB, Repo, Differ, DiffEvent, PullItemType, Remote};
use rusqlite::Connection;
use sqlite_commit_store::SqliteCommitGraphStore;
use s3_store::S3BlockStore;
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
    /// Initialize a new prolly repository
    Init {
        /// Path to prolly repository (will be created)
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Default author for commits
        #[arg(short, long, default_value = "prolly-cli")]
        author: String,
    },

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

    /// Get a value by key
    Get {
        /// Key to retrieve
        key: String,

        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Ref or commit to read from (default: HEAD)
        #[arg(long)]
        from: Option<String>,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Set a key-value pair and create a commit
    Set {
        /// Key to set
        key: String,

        /// Value to set
        value: String,

        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Commit message (default: "Set {key} = {value}")
        #[arg(short, long)]
        message: Option<String>,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Dump all key-value pairs from a ref or HEAD
    Dump {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Ref or commit to read from (default: HEAD)
        #[arg(long)]
        from: Option<String>,

        /// Optional key prefix to filter dump
        #[arg(short, long)]
        prefix: Option<String>,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Diff two commits or refs
    Diff {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Old ref/commit (default: HEAD~)
        #[arg(long)]
        old: Option<String>,

        /// New ref/commit (default: HEAD)
        #[arg(long)]
        new: Option<String>,

        /// Optional key prefix to filter diff results
        #[arg(short, long)]
        prefix: Option<String>,

        /// Maximum number of diff events to display
        #[arg(short, long)]
        limit: Option<usize>,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Print tree structure for a ref or commit
    PrintTree {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Ref or commit to visualize (default: HEAD)
        #[arg(long)]
        from: Option<String>,

        /// Verbose mode - show all leaf node values
        #[arg(short, long)]
        verbose: bool,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Push commits and nodes to a remote S3 store
    Push {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Remote name in config file (default: origin)
        #[arg(short = 'n', long, default_value = "origin")]
        remote_name: String,

        /// Number of parallel upload threads (default: 50)
        #[arg(short, long, default_value = "50")]
        threads: usize,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },

    /// Pull commits and nodes from a remote S3 store
    Pull {
        /// Path to prolly repository
        #[arg(short, long, default_value = ".prolly")]
        repo: PathBuf,

        /// Remote name in config file (default: origin)
        #[arg(short = 'n', long, default_value = "origin")]
        remote_name: String,

        /// Number of parallel download threads (default: 50)
        #[arg(short, long, default_value = "50")]
        threads: usize,

        /// Cache size for block store
        #[arg(short, long, default_value = "10000")]
        cache_size: usize,
    },
}

fn init_repo(repo_path: PathBuf, author: String) -> anyhow::Result<()> {
    // Check if repository already exists
    if repo_path.exists() {
        return Err(anyhow::anyhow!(
            "Repository already exists at {}",
            repo_path.display()
        ));
    }

    println!("Initializing prolly repository at {}", repo_path.display());

    // Create directory structure
    std::fs::create_dir_all(&repo_path)?;
    let store_path = repo_path.join("blocks");
    let commit_store_path = repo_path.join("commits.db");

    // Create stores
    let store = Arc::new(CachedFSBlockStore::new(&store_path, 1000)?);
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path)?);

    // Initialize empty repo with initial commit
    let repo = Repo::init_empty(store, commit_store, author.clone())?;

    // Get the initial commit info
    let (head_commit, ref_name) = repo.get_head()?;

    println!("Initialized empty prolly repository in {}", repo_path.display());
    if let Some(commit) = head_commit {
        let commit_hash = commit.compute_hash();
        println!("Initial commit: {}", hex::encode(&commit_hash[..8]));
    }
    println!("Branch: {}", ref_name);
    println!("Author: {}", author);

    Ok(())
}

fn open_repo(repo_path: &PathBuf, cache_size: usize) -> anyhow::Result<Repo> {
    if !repo_path.exists() {
        return Err(anyhow::anyhow!(
            "Repository not found at {}. Run 'prolly init' first.",
            repo_path.display()
        ));
    }

    let commit_db = repo_path.join("commits.db");
    if !commit_db.exists() {
        return Err(anyhow::anyhow!(
            "Repository at {} is not initialized. Run 'prolly init' first.",
            repo_path.display()
        ));
    }

    let store_path = repo_path.join("blocks");
    let store = Arc::new(CachedFSBlockStore::new(&store_path, cache_size)?);
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_db)?);
    let repo = Repo::new(store, commit_store, "prolly-cli".to_string());

    Ok(repo)
}

fn import_sqlite(
    sqlite_path: PathBuf,
    repo_path: PathBuf,
    batch_size: usize,
    cache_size: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    println!("Importing SQLite database: {}", sqlite_path.display());
    println!("Target repository: {}", repo_path.display());
    println!("Batch size: {}", batch_size);
    println!("Cache size: {}", cache_size);
    println!();

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

    // Create database
    let mut db = DB::new(repo.block_store.clone(), 0.01, 42);

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

        // Import data in batches - stream rows instead of loading all into memory
        let column_list = columns.join(", ");
        let query = format!("SELECT {} FROM {}", column_list, table_name);

        let mut stmt = sqlite_conn.prepare(&query)?;
        let mut rows_query = stmt.query([])?;

        let table_start = Instant::now();

        if batch_size == 0 {
            // Single batch mode: collect all rows and insert once for optimal tree structure
            let mut all_rows = Vec::new();
            while let Some(row) = rows_query.next()? {
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
                all_rows.push(values);
            }

            total_rows += all_rows.len();
            db.insert_rows(table_name, all_rows, verbose);
        } else {
            // Batched mode: insert in chunks for memory efficiency
            let mut row_buffer = Vec::new();
            while let Some(row) = rows_query.next()? {
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
                row_buffer.push(values);

                // Insert batch when full
                if row_buffer.len() >= batch_size {
                    let batch_len = row_buffer.len();
                    db.insert_rows(table_name, row_buffer, verbose);
                    total_rows += batch_len;
                    row_buffer = Vec::new();
                }
            }

            // Insert remaining rows
            if !row_buffer.is_empty() {
                let batch_len = row_buffer.len();
                db.insert_rows(table_name, row_buffer, verbose);
                total_rows += batch_len;
            }
        }

        let table_elapsed = table_start.elapsed();
        let rows_per_sec = total_rows as f64 / table_elapsed.as_secs_f64();

        println!(
            "Imported {} rows in {:.2}s ({:.0} rows/sec)",
            total_rows,
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
    println!("Total nodes: {}", repo.block_store.count_nodes()?);
    println!();

    // Create a commit for the imported data
    let root_hash = db.get_root_hash();
    let commit_message = format!(
        "Import from {}: {} rows from {} tables",
        sqlite_path.display(),
        total_rows,
        table_names.len()
    );
    let commit = repo.commit(&root_hash, &commit_message, None, None, None)?;
    let commit_hash = commit.compute_hash();

    println!("Created commit: {}", hex::encode(&commit_hash[..8]));
    println!("Message: {}", commit_message);

    Ok(())
}

fn gc_repo(repo_path: PathBuf, dry_run: bool, cache_size: usize) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    println!("Garbage collecting repository: {}", repo_path.display());
    println!("Mode: {}", if dry_run { "dry-run" } else { "live" });

    println!();
    println!("Finding reachable tree roots...");
    let tree_roots = repo.get_reachable_tree_roots()?;
    println!("Found {} reachable tree roots", tree_roots.len());

    println!();
    println!("Running garbage collection...");
    let start = Instant::now();
    let stats = garbage_collect(repo.block_store.as_ref(), &tree_roots, dry_run);
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
    let repo = open_repo(&repo_path, cache_size)?;

    if let Some(branch_name) = name {
        // Create a new branch
        let from_commit = if let Some(from_ref) = from {
            Some(repo.resolve_ref(&from_ref)?.ok_or_else(|| {
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
        let (head_commit, head_ref) = repo.get_head()?;
        let branches = repo.list_branches()?;

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
    let repo = open_repo(&repo_path, cache_size)?;

    repo.checkout(&branch)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Switched to branch: {}", branch);

    // Show commit info
    let (commit, _) = repo.get_head()?;
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
    let repo = open_repo(&repo_path, cache_size)?;

    let commits = repo.log(start.as_deref(), max_count)?;

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

fn get_key(
    repo_path: PathBuf,
    key: String,
    from: Option<String>,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Determine which commit to read from
    let (commit, ref_display) = if let Some(ref_name) = from {
        let commit_hash = repo.resolve_ref(&ref_name)?.ok_or_else(|| {
            anyhow::anyhow!("Ref '{}' not found", ref_name)
        })?;
        let commit = repo.commit_graph_store.get_commit(&commit_hash)?.ok_or_else(|| {
            anyhow::anyhow!("Commit not found")
        })?;
        (commit, ref_name)
    } else {
        let (head_commit, ref_name) = repo.get_head()?;
        let commit = head_commit.ok_or_else(|| {
            anyhow::anyhow!("No commits in repository")
        })?;
        (commit, format!("HEAD ({})", ref_name))
    };

    // Load tree with pattern and seed from commit
    let mut tree = ProllyTree::new(commit.pattern, commit.seed as u32, Some(repo.block_store.clone()));

    // Try to load the root node. If it doesn't exist (empty tree), use a new empty tree
    if let Ok(Some(root_node)) = repo.block_store.get_node(&commit.tree_root) {
        tree.root = (*root_node).clone();
    }
    // else: tree already has an empty root from ProllyTree::new()

    // Search for the key using cursor
    let key_bytes = key.as_bytes().to_vec();
    let mut cursor = TreeCursor::new(repo.block_store.as_ref(), tree.get_root_hash(), Some(&key_bytes));

    if let Some((found_key, value)) = cursor.next() {
        if found_key.as_ref() == key_bytes.as_slice() {
            println!("================================================================================");
            println!("GET from {}", ref_display);
            println!("================================================================================");
            println!("Key:   {}", key);
            println!("Value: {}", String::from_utf8_lossy(&value));
            println!("================================================================================");
            return Ok(());
        }
    }

    println!("Key '{}' not found", key);
    Ok(())
}

fn set_key(
    repo_path: PathBuf,
    key: String,
    value: String,
    message: Option<String>,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Get current HEAD
    let (head_commit, ref_name) = repo.get_head()?;
    let head_commit = head_commit.ok_or_else(|| {
        anyhow::anyhow!("No commits in repository. Use 'prolly init' first.")
    })?;

    // Load current tree with pattern and seed from commit
    let mut tree = ProllyTree::new(head_commit.pattern, head_commit.seed as u32, Some(repo.block_store.clone()));

    // Try to load the root node. If it doesn't exist (empty tree), use a new empty tree
    if let Ok(Some(root_node)) = repo.block_store.get_node(&head_commit.tree_root) {
        tree.root = (*root_node).clone();
    }
    // else: tree already has an empty root from ProllyTree::new()

    // Insert the key-value pair
    let key_bytes = key.as_bytes().to_vec();
    let value_bytes = value.as_bytes().to_vec();
    tree.insert_batch(vec![(key_bytes, value_bytes)], false);

    // Get new root hash
    let new_root_hash = tree.get_root_hash();

    // Create commit
    let commit_message = message.unwrap_or_else(|| format!("Set {} = {}", key, value));
    let commit = repo.commit(
        &new_root_hash,
        &commit_message,
        None,
        Some(head_commit.pattern),
        Some(head_commit.seed as u32),
    )?;
    let commit_hash = commit.compute_hash();

    println!("================================================================================");
    println!("SET COMPLETE");
    println!("================================================================================");
    println!("Key:      {}", key);
    println!("Value:    {}", value);
    println!("Commit:   {}", hex::encode(&commit_hash[..8]));
    println!("Message:  {}", commit_message);
    println!("Branch:   {}", ref_name);
    println!("================================================================================");

    Ok(())
}

fn dump_cmd(
    repo_path: PathBuf,
    from: Option<String>,
    prefix: Option<String>,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Determine which commit to read from
    let (commit, ref_display) = if let Some(ref_name) = from {
        let commit_hash = repo.resolve_ref(&ref_name)?.ok_or_else(|| {
            anyhow::anyhow!("Ref '{}' not found", ref_name)
        })?;
        let commit = repo.commit_graph_store.get_commit(&commit_hash)?.ok_or_else(|| {
            anyhow::anyhow!("Commit not found")
        })?;
        (commit, ref_name)
    } else {
        let (head_commit, ref_name) = repo.get_head()?;
        let commit = head_commit.ok_or_else(|| {
            anyhow::anyhow!("No commits in repository")
        })?;
        (commit, format!("HEAD ({})", ref_name))
    };

    println!("================================================================================");
    println!("Dumping from {}", ref_display);
    println!("================================================================================");
    println!("Tree root hash: {}", hex::encode(&commit.tree_root));

    // Load tree with pattern and seed from commit
    let mut tree = ProllyTree::new(commit.pattern, commit.seed as u32, Some(repo.block_store.clone()));

    // Try to load the root node. If it doesn't exist (empty tree), use a new empty tree
    if let Ok(Some(root_node)) = repo.block_store.get_node(&commit.tree_root) {
        tree.root = (*root_node).clone();
    }

    // Get prefix bytes if provided
    let prefix_bytes = prefix.as_ref().map(|p| p.as_bytes());
    let prefix_str = prefix.as_deref().unwrap_or("");

    println!("\nKeys with prefix: '{}'", prefix_str);
    println!("--------------------------------------------------------------------------------");

    // Use TreeCursor to iterate with optional prefix
    let root_hash = tree.get_root_hash();
    let mut cursor = TreeCursor::new(repo.block_store.as_ref(), root_hash, prefix_bytes);
    let mut count = 0;

    while let Some((key, value)) = cursor.next() {
        // If we have a prefix, check if we've moved past it
        if let Some(prefix_bytes) = prefix_bytes {
            if !key.starts_with(prefix_bytes) {
                break;
            }
        }

        // Print key-value pair
        let key_str = String::from_utf8_lossy(&key);
        let value_str = String::from_utf8_lossy(&value);
        println!("{} => {}", key_str, value_str);
        count += 1;
    }

    println!("================================================================================");
    println!("Total: {} keys found", count);

    Ok(())
}

fn diff_cmd(
    repo_path: PathBuf,
    old: Option<String>,
    new: Option<String>,
    prefix: Option<String>,
    limit: Option<usize>,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Default to HEAD~ vs HEAD
    let old_ref = old.unwrap_or_else(|| "HEAD~".to_string());
    let new_ref = new.unwrap_or_else(|| "HEAD".to_string());

    // Resolve old ref
    let old_commit_hash = repo.resolve_ref(&old_ref)?.ok_or_else(|| {
        anyhow::anyhow!("Ref '{}' not found", old_ref)
    })?;
    let old_commit = repo.commit_graph_store.get_commit(&old_commit_hash)?.ok_or_else(|| {
        anyhow::anyhow!("Commit not found")
    })?;

    // Resolve new ref
    let new_commit_hash = repo.resolve_ref(&new_ref)?.ok_or_else(|| {
        anyhow::anyhow!("Ref '{}' not found", new_ref)
    })?;
    let new_commit = repo.commit_graph_store.get_commit(&new_commit_hash)?.ok_or_else(|| {
        anyhow::anyhow!("Commit not found")
    })?;

    println!("================================================================================");
    println!("DIFF: Comparing two commits");
    println!("================================================================================");
    println!("Old: {} (tree: {}...)", old_ref, hex::encode(&old_commit.tree_root[..8]));
    println!("New: {} (tree: {}...)", new_ref, hex::encode(&new_commit.tree_root[..8]));
    if let Some(ref p) = prefix {
        println!("Prefix: {}", p);
    }

    if old_commit.tree_root == new_commit.tree_root {
        println!("\nTrees are identical (same root hash)");
        return Ok(());
    }

    // Create Differ instance
    let mut differ = Differ::new(repo.block_store.as_ref());

    println!("\nDiff events (old -> new):");
    println!("--------------------------------------------------------------------------------");

    // Get prefix bytes if provided
    let prefix_bytes = prefix.as_ref().map(|p| p.as_bytes());

    // Run diff
    let events = differ.diff(&old_commit.tree_root, &new_commit.tree_root, prefix_bytes);

    let mut event_count = 0;
    let mut added_count = 0;
    let mut deleted_count = 0;
    let mut modified_count = 0;

    for event in &events {
        event_count += 1;

        if limit.is_none() || event_count <= limit.unwrap() {
            match event {
                DiffEvent::Added(added) => {
                    let key_str = String::from_utf8_lossy(&added.key);
                    let value_str = String::from_utf8_lossy(&added.value);
                    println!("+ {} = {}", key_str, value_str);
                    added_count += 1;
                }
                DiffEvent::Deleted(deleted) => {
                    let key_str = String::from_utf8_lossy(&deleted.key);
                    let value_str = String::from_utf8_lossy(&deleted.old_value);
                    println!("- {} = {}", key_str, value_str);
                    deleted_count += 1;
                }
                DiffEvent::Modified(modified) => {
                    let key_str = String::from_utf8_lossy(&modified.key);
                    let old_str = String::from_utf8_lossy(&modified.old_value);
                    let new_str = String::from_utf8_lossy(&modified.new_value);
                    println!("M {}: {} -> {}", key_str, old_str, new_str);
                    modified_count += 1;
                }
            }
        } else {
            // Just count without printing
            match event {
                DiffEvent::Added(_) => added_count += 1,
                DiffEvent::Deleted(_) => deleted_count += 1,
                DiffEvent::Modified(_) => modified_count += 1,
            }
        }
    }

    println!("--------------------------------------------------------------------------------");
    println!("\nDiff Summary:");
    println!("  Added:    {}", added_count);
    println!("  Deleted:  {}", deleted_count);
    println!("  Modified: {}", modified_count);
    println!("  Total:    {}", event_count);

    if let Some(limit) = limit {
        if event_count > limit {
            println!("\n(showing first {}, {} more events omitted)", limit, event_count - limit);
        }
    }

    // Print diff statistics
    let diff_stats = differ.get_stats();
    println!("\nDiff Algorithm Statistics:");
    println!("  Subtrees skipped (identical hashes): {}", diff_stats.subtrees_skipped);
    println!("  Nodes compared:                      {}", diff_stats.nodes_compared);

    Ok(())
}

fn print_tree_cmd(
    repo_path: PathBuf,
    from: Option<String>,
    verbose: bool,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Default to HEAD if no ref provided
    let ref_name = from.unwrap_or_else(|| "HEAD".to_string());

    // Resolve ref to commit
    let commit_hash = repo.resolve_ref(&ref_name)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .ok_or_else(|| anyhow::anyhow!("Ref '{}' not found", ref_name))?;
    let commit = repo.commit_graph_store.get_commit(&commit_hash)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .ok_or_else(|| anyhow::anyhow!("Commit not found"))?;

    println!("================================================================================");
    println!("TREE STRUCTURE");
    println!("================================================================================");
    println!("Ref:       {}", ref_name);
    println!("Commit:    {}", hex::encode(&commit_hash[..8]));
    println!("Tree root: {}", hex::encode(&commit.tree_root));
    if !verbose {
        println!("Mode:      compact (use --verbose to show all leaf values)");
    }

    // Create tree with pattern and seed from commit
    let mut tree = ProllyTree::new(commit.pattern, commit.seed as u32, Some(repo.block_store.clone()));

    // Try to load the root node
    match repo.block_store.get_node(&commit.tree_root) {
        Ok(Some(root_node)) => {
            tree.root = (*root_node).clone();
        }
        Ok(None) => {
            println!("\nError: Tree root not found in store");
            return Ok(());
        }
        Err(e) => {
            return Err(anyhow::anyhow!("Error loading tree root: {}", e));
        }
    }

    // Print the tree structure
    let label = format!("root={}", hex::encode(&commit.tree_root[..8]));
    print_tree_recursive(
        &tree.root,
        &commit.tree_root,
        repo.block_store.as_ref(),
        &label,
        "",
        true,
        verbose,
    );

    Ok(())
}

fn print_tree_recursive(
    node: &prolly_core::Node,
    node_hash: &[u8],
    store: &dyn BlockStore,
    _label: &str,
    prefix: &str,
    is_last: bool,
    verbose: bool,
) {
    let branch = if is_last { "└── " } else { "├── " };
    let hash_str = hex::encode(&node_hash[..8]);

    if node.is_leaf {
        if verbose {
            // Show all key-value pairs
            let mut data = Vec::new();
            for i in 0..node.keys.len() {
                let key = String::from_utf8_lossy(&node.keys[i]);
                let value = String::from_utf8_lossy(&node.values[i]);
                data.push(format!("({}, {})", key, value));
            }
            println!("{}{}LEAF #{}: [{}]", prefix, branch, hash_str, data.join(", "));
        } else {
            // Show only first and last keys, and the count
            let count = node.keys.len();
            if count == 0 {
                println!("{}{}LEAF #{}: (empty)", prefix, branch, hash_str);
            } else if count == 1 {
                let key = String::from_utf8_lossy(&node.keys[0]);
                println!("{}{}LEAF #{}: [{}] (1 key)", prefix, branch, hash_str, key);
            } else {
                let first_key = String::from_utf8_lossy(&node.keys[0]);
                let last_key = String::from_utf8_lossy(&node.keys[count - 1]);
                println!("{}{}LEAF #{}: [{} ... {}] ({} keys)", prefix, branch, hash_str, first_key, last_key, count);
            }
        }
    } else {
        // Internal node - print keys
        let keys_str: Vec<String> = node.keys.iter()
            .map(|k| String::from_utf8_lossy(k).to_string())
            .collect();
        println!("{}{}INTERNAL #{}: keys=[{}]", prefix, branch, hash_str, keys_str.join(", "));

        // Print children
        let extension = if is_last { "    " } else { "│   " };
        for (i, child_hash) in node.values.iter().enumerate() {
            let hash_vec = child_hash.to_vec();
            if let Ok(Some(child_node)) = store.get_node(&hash_vec) {
                let child_is_last = i == node.values.len() - 1;
                print_tree_recursive(
                    &child_node,
                    child_hash,
                    store,
                    "",
                    &format!("{}{}", prefix, extension),
                    child_is_last,
                    verbose,
                );
            }
        }
    }
}

fn push_cmd(
    repo_path: PathBuf,
    remote_name: String,
    threads: usize,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Load config from repo
    let config_path = repo_path.join("config.toml");
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Config file not found at {}. Create a config.toml with remote configuration.",
            config_path.display()
        ));
    }

    // Create S3 remote from config
    let remote = Arc::new(S3BlockStore::from_config(&config_path, &remote_name)?);

    println!("================================================================================");
    println!("PUSH to {}", remote.url());
    println!("================================================================================");
    let (_, ref_name) = repo.get_head().map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Branch: {}", ref_name);
    println!("Threads: {}", threads);

    let start = Instant::now();

    // Push with progress callback
    let pushed = repo.push(
        remote.clone(),
        threads,
        Some(Box::new(|done, total| {
            if done % 100 == 0 || done == total {
                print!("\rPushing nodes: {}/{} ({:.1}%)", done, total, (done as f64 / total as f64) * 100.0);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        })),
    ).map_err(|e| anyhow::anyhow!("{}", e))?;

    let elapsed = start.elapsed();

    println!();
    println!("================================================================================");
    println!("Push complete!");
    println!("Nodes pushed: {}", pushed);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    if pushed > 0 {
        println!("Rate: {:.0} nodes/sec", pushed as f64 / elapsed.as_secs_f64());
    }
    println!("================================================================================");

    Ok(())
}

fn pull_cmd(
    repo_path: PathBuf,
    remote_name: String,
    threads: usize,
    cache_size: usize,
) -> anyhow::Result<()> {
    let repo = open_repo(&repo_path, cache_size)?;

    // Load config from repo
    let config_path = repo_path.join("config.toml");
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Config file not found at {}. Create a config.toml with remote configuration.",
            config_path.display()
        ));
    }

    // Create S3 remote from config
    let remote = Arc::new(S3BlockStore::from_config(&config_path, &remote_name)?);

    println!("================================================================================");
    println!("PULL from {}", remote.url());
    println!("================================================================================");
    let (_, ref_name) = repo.get_head().map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Branch: {}", ref_name);
    println!("Threads: {}", threads);

    let start = Instant::now();

    // Track counts for summary
    let commits_pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let nodes_pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let commits_pulled_clone = commits_pulled.clone();
    let nodes_pulled_clone = nodes_pulled.clone();

    // Pull with progress callback
    let total = repo.pull(
        remote.clone(),
        threads,
        Some(Box::new(move |progress| {
            // Count item types
            match progress.item_type {
                PullItemType::Commit => {
                    commits_pulled_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                PullItemType::Node => {
                    nodes_pulled_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Display progress (similar to Python version)
            print!("\rDone: {} | In progress: {} | Pending: {}  ",
                   progress.done, progress.in_progress, progress.pending);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        })),
    ).map_err(|e| anyhow::anyhow!("{}", e))?;

    println!(); // Newline after progress display

    let elapsed = start.elapsed();

    let commits = commits_pulled.load(std::sync::atomic::Ordering::Relaxed);
    let nodes = nodes_pulled.load(std::sync::atomic::Ordering::Relaxed);

    println!("================================================================================");
    println!("Pull complete!");
    println!("Commits pulled: {}", commits);
    println!("Nodes pulled: {}", nodes);
    println!("Total items: {}", total);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    if total > 0 {
        println!("Rate: {:.0} items/sec", total as f64 / elapsed.as_secs_f64());
    }

    // Show updated HEAD
    let (head_commit, ref_name) = repo.get_head().map_err(|e| anyhow::anyhow!("{}", e))?;
    if let Some(commit) = head_commit {
        println!();
        println!("HEAD -> {}", ref_name);
        println!("Commit: {}", hex::encode(&commit.compute_hash()[..8]));
        println!("Message: {}", commit.message);
    }
    println!("================================================================================");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { repo, author } => init_repo(repo, author)?,
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
        Commands::Get {
            key,
            repo,
            from,
            cache_size,
        } => get_key(repo, key, from, cache_size)?,
        Commands::Set {
            key,
            value,
            repo,
            message,
            cache_size,
        } => set_key(repo, key, value, message, cache_size)?,
        Commands::Dump {
            repo,
            from,
            prefix,
            cache_size,
        } => dump_cmd(repo, from, prefix, cache_size)?,
        Commands::Diff {
            repo,
            old,
            new,
            prefix,
            limit,
            cache_size,
        } => diff_cmd(repo, old, new, prefix, limit, cache_size)?,
        Commands::PrintTree {
            repo,
            from,
            verbose,
            cache_size,
        } => print_tree_cmd(repo, from, verbose, cache_size)?,
        Commands::Push {
            repo,
            remote_name,
            threads,
            cache_size,
        } => push_cmd(repo, remote_name, threads, cache_size)?,
        Commands::Pull {
            repo,
            remote_name,
            threads,
            cache_size,
        } => pull_cmd(repo, remote_name, threads, cache_size)?,
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

        Repo::init_empty(store, commit_store, "test@example.com".to_string()).unwrap()
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

        // Initialize the repo first
        init_repo(repo_path.clone(), "test@example.com".to_string()).unwrap();

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

        repo.commit(&tree_root, "Initial commit", None, None, None).unwrap();

        // Create a new branch
        let head_commit = repo.resolve_ref("main").unwrap().unwrap();
        repo.create_branch("feature", Some(&head_commit)).unwrap();

        // List branches
        let branches = repo.list_branches().unwrap();
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
        repo.commit(&tree_root1, "First commit", None, None, None).unwrap();

        // Create a branch and switch to it
        let main_commit = repo.resolve_ref("main").unwrap().unwrap();
        repo.create_branch("develop", Some(&main_commit)).unwrap();
        repo.checkout("develop").unwrap();

        // Verify we're on develop
        let (_, head_ref) = repo.get_head().unwrap();
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
        repo.commit(&tree_root2, "Second commit on develop", None, None, None).unwrap();

        // Switch back to main
        repo.checkout("main").unwrap();
        let (_, head_ref) = repo.get_head().unwrap();
        assert_eq!(head_ref, "main");

        // Main should still point to first commit
        let (main_commit, _) = repo.get_head().unwrap();
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
            repo.commit(&tree_root, &format!("Commit {}", i), None, None, None).unwrap();
        }

        // Get log from HEAD
        let commits = repo.log(None, None).unwrap();
        assert_eq!(commits.len(), 4); // 3 commits + initial commit

        // Commits should be in reverse chronological order
        assert_eq!(commits[0].1.message, "Commit 3");
        assert_eq!(commits[1].1.message, "Commit 2");
        assert_eq!(commits[2].1.message, "Commit 1");
        assert_eq!(commits[3].1.message, "Initial commit");

        // Test with max_count
        let commits = repo.log(None, Some(2)).unwrap();
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
        repo.commit(&tree_root, "Test commit", None, None, None).unwrap();

        // Run GC - should find no garbage since all nodes are reachable
        let tree_roots = repo.get_reachable_tree_roots().unwrap();
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
        repo.commit(&tree_root1, "First commit", None, None, None).unwrap();

        // Create an orphaned tree (not committed)
        let mut orphan_tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        orphan_tree.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);
        let _orphan_root = orphan_tree.get_root_hash();

        let nodes_before = repo.block_store.count_nodes().unwrap();

        // Run GC - should find the orphaned nodes
        let tree_roots = repo.get_reachable_tree_roots().unwrap();
        let stats = garbage_collect(repo.block_store.as_ref(), &tree_roots, false);

        assert!(stats.garbage_nodes > 0, "Should have found garbage nodes");

        let nodes_after = repo.block_store.count_nodes().unwrap();
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
        repo.commit(&tree_root1, "First commit", None, None, None).unwrap();

        let mut orphan_tree = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        orphan_tree.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);

        let nodes_before = repo.block_store.count_nodes().unwrap();

        // Run GC in dry-run mode
        let tree_roots = repo.get_reachable_tree_roots().unwrap();
        let stats = garbage_collect(repo.block_store.as_ref(), &tree_roots, true);

        let nodes_after = repo.block_store.count_nodes().unwrap();

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
            let commit = repo.commit(&tree_root, &format!("Commit {}", i), None, None, None).unwrap();
            let commit_hash = commit.compute_hash();
            commits.push(commit_hash);
        }

        // Test HEAD resolves to latest commit
        let head = repo.resolve_ref("HEAD").unwrap().unwrap();
        assert_eq!(head, commits[2]); // commits[2] is "Commit 3"

        // Test HEAD~1 resolves to parent
        let head_1 = repo.resolve_ref("HEAD~1").unwrap().unwrap();
        assert_eq!(head_1, commits[1]);

        // Test HEAD~2
        let head_2 = repo.resolve_ref("HEAD~2").unwrap().unwrap();
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
        repo.commit(&tree_root, "Test", None, None, None).unwrap();

        // Use the CLI function to create a branch
        let result = branch_cmd(repo_path.clone(), Some("feature".to_string()), None, 100);
        assert!(result.is_ok());

        // Verify branch was created
        let branches = repo.list_branches().unwrap();
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
        repo.commit(&tree_root, "Test", None, None, None).unwrap();

        let main_commit = repo.resolve_ref("main").unwrap().unwrap();
        repo.create_branch("develop", Some(&main_commit)).unwrap();

        // Use the CLI function to checkout
        let result = checkout_cmd(repo_path.clone(), "develop".to_string(), 100);
        assert!(result.is_ok());

        // Verify we switched
        let (_, head_ref) = repo.get_head().unwrap();
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
        repo.commit(&tree_root, "Test", None, None, None).unwrap();

        let mut orphan = ProllyTree::new(0.01, 42, Some(repo.block_store.clone()));
        orphan.insert_batch(vec![(b"orphan".to_vec(), b"data".to_vec())], false);

        let nodes_before = repo.block_store.count_nodes().unwrap();

        // Run GC via CLI function
        let result = gc_repo(repo_path.clone(), true, 100);
        assert!(result.is_ok());

        // Verify dry run didn't remove nodes
        let nodes_after = repo.block_store.count_nodes().unwrap();
        assert_eq!(nodes_before, nodes_after);
    }

    #[test]
    fn test_init_creates_repo() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("test_repo");

        // Init should succeed
        let result = init_repo(repo_path.clone(), "test@example.com".to_string());
        assert!(result.is_ok());

        // Repo directory should exist
        assert!(repo_path.exists());
        assert!(repo_path.join("blocks").exists());
        assert!(repo_path.join("commits.db").exists());

        // Should fail if run again
        let result = init_repo(repo_path.clone(), "test@example.com".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_open_repo_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("nonexistent");

        // Should fail for nonexistent repo
        let result = open_repo(&repo_path, 100);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("Run 'prolly init'"));
    }

    #[test]
    fn test_commands_require_init() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("uninit_repo");

        // All commands should fail without init
        assert!(gc_repo(repo_path.clone(), true, 100).is_err());
        assert!(branch_cmd(repo_path.clone(), None, None, 100).is_err());
        assert!(checkout_cmd(repo_path.clone(), "main".to_string(), 100).is_err());
        assert!(log_cmd(repo_path.clone(), None, None, 100).is_err());
        assert!(get_key(repo_path.clone(), "key".to_string(), None, 100).is_err());
        assert!(set_key(repo_path.clone(), "key".to_string(), "value".to_string(), None, 100).is_err());
    }

    #[test]
    fn test_set_and_get_key() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Set a key
        let result = set_key(
            repo_path.clone(),
            "mykey".to_string(),
            "myvalue".to_string(),
            None,
            100,
        );
        if let Err(e) = &result {
            eprintln!("Set error: {}", e);
        }
        assert!(result.is_ok());

        // Get the key back
        let result = get_key(repo_path.clone(), "mykey".to_string(), None, 100);
        assert!(result.is_ok());

        // Verify a commit was created
        let commits = repo.log(None, None).unwrap();
        assert!(commits.len() >= 2); // Initial commit + set commit
        assert!(commits[0].1.message.contains("Set mykey = myvalue"));
    }

    #[test]
    fn test_set_creates_commit() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        let commits_before = repo.log(None, None).unwrap().len();

        // Set a key with custom message
        let result = set_key(
            repo_path.clone(),
            "testkey".to_string(),
            "testvalue".to_string(),
            Some("Custom message".to_string()),
            100,
        );
        assert!(result.is_ok());

        // Verify a new commit was created
        let commits_after = repo.log(None, None).unwrap();
        assert_eq!(commits_after.len(), commits_before + 1);
        assert_eq!(commits_after[0].1.message, "Custom message");
    }

    #[test]
    fn test_get_nonexistent_key() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        create_test_repo(&repo_path);

        // Try to get a key that doesn't exist
        let result = get_key(repo_path, "nonexistent".to_string(), None, 100);
        assert!(result.is_ok()); // Should succeed but print "not found"
    }

    #[test]
    fn test_set_multiple_keys() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Set multiple keys
        for i in 1..=3 {
            let result = set_key(
                repo_path.clone(),
                format!("key{}", i),
                format!("value{}", i),
                None,
                100,
            );
            assert!(result.is_ok());
        }

        // Verify all commits were created
        let commits = repo.log(None, None).unwrap();
        assert_eq!(commits.len(), 4); // Initial + 3 sets
        assert!(commits[0].1.message.contains("Set key3 = value3"));
        assert!(commits[1].1.message.contains("Set key2 = value2"));
        assert!(commits[2].1.message.contains("Set key1 = value1"));
    }

    #[test]
    fn test_get_from_different_ref() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Set a key
        set_key(
            repo_path.clone(),
            "mykey".to_string(),
            "original".to_string(),
            None,
            100,
        )
        .unwrap();

        // Create a branch
        let commit = repo.resolve_ref("main").unwrap().unwrap();
        repo.create_branch("backup", Some(&commit)).unwrap();

        // Update the key
        set_key(
            repo_path.clone(),
            "mykey".to_string(),
            "updated".to_string(),
            None,
            100,
        )
        .unwrap();

        // Get from backup branch should have original value
        // (We can't easily test the output, but we can verify it doesn't error)
        let result = get_key(
            repo_path.clone(),
            "mykey".to_string(),
            Some("backup".to_string()),
            100,
        );
        assert!(result.is_ok());

        // Get from HEAD should have updated value
        let result = get_key(repo_path, "mykey".to_string(), None, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dump_cmd() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        create_test_repo(&repo_path);

        // Set some keys
        set_key(
            repo_path.clone(),
            "key1".to_string(),
            "value1".to_string(),
            None,
            100,
        )
        .unwrap();

        set_key(
            repo_path.clone(),
            "key2".to_string(),
            "value2".to_string(),
            None,
            100,
        )
        .unwrap();

        set_key(
            repo_path.clone(),
            "prefix/key3".to_string(),
            "value3".to_string(),
            None,
            100,
        )
        .unwrap();

        // Test dump without prefix
        let result = dump_cmd(repo_path.clone(), None, None, 100);
        assert!(result.is_ok());

        // Test dump with prefix
        let result = dump_cmd(repo_path.clone(), None, Some("prefix/".to_string()), 100);
        assert!(result.is_ok());

        // Test dump from specific ref
        let result = dump_cmd(repo_path, Some("HEAD".to_string()), None, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_diff_cmd() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Set initial key
        set_key(
            repo_path.clone(),
            "key1".to_string(),
            "value1".to_string(),
            None,
            100,
        )
        .unwrap();

        // Create a branch at this point
        let commit = repo.resolve_ref("main").unwrap().unwrap();
        repo.create_branch("old", Some(&commit)).unwrap();

        // Modify the key
        set_key(
            repo_path.clone(),
            "key1".to_string(),
            "value2".to_string(),
            None,
            100,
        )
        .unwrap();

        // Add a new key
        set_key(
            repo_path.clone(),
            "key2".to_string(),
            "newvalue".to_string(),
            None,
            100,
        )
        .unwrap();

        // Test diff between old and HEAD
        let result = diff_cmd(
            repo_path.clone(),
            Some("old".to_string()),
            Some("HEAD".to_string()),
            None,
            None,
            100,
        );
        assert!(result.is_ok());

        // Test diff with limit
        let result = diff_cmd(
            repo_path.clone(),
            Some("old".to_string()),
            Some("HEAD".to_string()),
            None,
            Some(1),
            100,
        );
        assert!(result.is_ok());

        // Test diff with prefix
        let result = diff_cmd(
            repo_path,
            Some("old".to_string()),
            Some("HEAD".to_string()),
            Some("key1".to_string()),
            None,
            100,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_tree_cmd() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        create_test_repo(&repo_path);

        // Set some keys to create a non-empty tree
        set_key(
            repo_path.clone(),
            "key1".to_string(),
            "value1".to_string(),
            None,
            100,
        )
        .unwrap();

        set_key(
            repo_path.clone(),
            "key2".to_string(),
            "value2".to_string(),
            None,
            100,
        )
        .unwrap();

        // Test print tree in compact mode
        let result = print_tree_cmd(repo_path.clone(), None, false, 100);
        assert!(result.is_ok());

        // Test print tree in verbose mode
        let result = print_tree_cmd(repo_path.clone(), Some("HEAD".to_string()), true, 100);
        assert!(result.is_ok());

        // Test print tree from specific ref
        let result = print_tree_cmd(repo_path, Some("main".to_string()), false, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_diff_identical_trees() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        let repo = create_test_repo(&repo_path);

        // Set a key
        set_key(
            repo_path.clone(),
            "key1".to_string(),
            "value1".to_string(),
            None,
            100,
        )
        .unwrap();

        // Create a branch - should be identical to main
        let commit = repo.resolve_ref("main").unwrap().unwrap();
        repo.create_branch("same", Some(&commit)).unwrap();

        // Diff identical trees should succeed
        let result = diff_cmd(
            repo_path,
            Some("main".to_string()),
            Some("same".to_string()),
            None,
            None,
            100,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_dump_empty_tree() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("repo");
        std::fs::create_dir_all(&repo_path).unwrap();

        create_test_repo(&repo_path);

        // Dump empty tree should succeed
        let result = dump_cmd(repo_path, None, None, 100);
        assert!(result.is_ok());
    }
}
