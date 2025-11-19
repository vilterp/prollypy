mod sqlite_commit_store;

use clap::{Parser, Subcommand};
use prolly_core::{garbage_collect, CachedFSBlockStore, DB, Repo};
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
        #[arg(short = 'n', long)]
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
            .collect::<Result<_, _>>()?;

        let columns: Vec<String> = table_info.iter().map(|(name, _, _)| name.clone()).collect();
        let types: Vec<String> = table_info.iter().map(|(_, typ, _)| typ.clone()).collect();

        // Get primary key
        let mut pk_columns: Vec<_> = table_info
            .iter()
            .enumerate()
            .filter(|(_, (_, _, pk))| *pk > 0)
            .map(|(idx, (name, _, pk))| (*pk, name.clone(), idx))
            .collect();

        pk_columns.sort_by_key(|(pk, _, _)| *pk);

        let primary_key: Vec<String> = if pk_columns.is_empty() {
            vec!["rowid".to_string()]
        } else {
            pk_columns.iter().map(|(_, name, _)| name.clone()).collect()
        };

        println!("Columns: {}", columns.join(", "));
        println!("Primary key: {}", primary_key.join(", "));

        // Get row count
        let row_count: i64 = sqlite_conn.query_row(
            &format!("SELECT COUNT(*) FROM {}", table_name),
            [],
            |row| row.get(0),
        )?;
        println!("Total rows: {}", row_count);

        if row_count == 0 {
            println!("Skipping empty table");
            println!();
            continue;
        }

        // Create table in DB
        db.create_table(
            table_name.clone(),
            columns.clone(),
            types.clone(),
            primary_key.clone(),
        );

        // Import rows
        let query = if primary_key == vec!["rowid"] {
            format!("SELECT rowid, * FROM {}", table_name)
        } else {
            format!("SELECT * FROM {}", table_name)
        };

        let mut stmt = sqlite_conn.prepare(&query)?;
        let column_count = stmt.column_count();

        let import_start = Instant::now();
        let mut batch = Vec::new();
        let mut batch_num = 0;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let mut values = Vec::new();
            for i in 0..column_count {
                let value: serde_json::Value = match row.get_ref(i)? {
                    rusqlite::types::ValueRef::Null => serde_json::Value::Null,
                    rusqlite::types::ValueRef::Integer(n) => serde_json::Value::Number(n.into()),
                    rusqlite::types::ValueRef::Real(f) => {
                        serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap())
                    }
                    rusqlite::types::ValueRef::Text(s) => {
                        serde_json::Value::String(String::from_utf8_lossy(s).to_string())
                    }
                    rusqlite::types::ValueRef::Blob(b) => {
                        serde_json::Value::String(hex::encode(b))
                    }
                };
                values.push(value);
            }

            batch.push(values);

            if batch.len() >= batch_size || batch.len() as i64 == row_count {
                batch_num += 1;
                let batch_rows = batch.len();
                db.insert_rows(table_name, batch, verbose);
                batch = Vec::new();

                if verbose {
                    println!("  Batch {}: {} rows", batch_num, batch_rows);
                }
            }
        }

        // Insert any remaining rows
        if !batch.is_empty() {
            batch_num += 1;
            let batch_rows = batch.len();
            db.insert_rows(table_name, batch, verbose);

            if verbose {
                println!("  Batch {}: {} rows", batch_num, batch_rows);
            }
        }

        let import_elapsed = import_start.elapsed();
        let rows_per_sec = row_count as f64 / import_elapsed.as_secs_f64();

        println!(
            "Imported {} rows in {:.2}s ({:.0} rows/sec)",
            row_count,
            import_elapsed.as_secs_f64(),
            rows_per_sec
        );

        // Get cache stats
        let cache_stats = store.get_cache_stats();
        println!(
            "Cache: {} hits, {} misses ({:.1}% hit rate)",
            cache_stats.cache_hits, cache_stats.cache_misses, cache_stats.hit_rate
        );

        total_rows += row_count;
        println!();
    }

    let total_elapsed = total_start.elapsed();
    let total_rows_per_sec = total_rows as f64 / total_elapsed.as_secs_f64();

    println!("=== Import Complete ===");
    println!("Total rows: {}", total_rows);
    println!("Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("Average: {:.0} rows/sec", total_rows_per_sec);

    // Get creation stats
    let creation_stats = store.get_creation_stats();
    println!("Nodes created: {}", creation_stats.total_leaves_created + creation_stats.total_internals_created);
    println!("  Leaves: {}", creation_stats.total_leaves_created);
    println!("  Internals: {}", creation_stats.total_internals_created);

    // Create commit
    println!();
    println!("Creating commit...");
    let commit_store_path = repo_path.join("commits.db");
    let commit_store = Arc::new(SqliteCommitGraphStore::new(&commit_store_path)?);
    let repo = Repo::new(store.clone(), commit_store.clone(), "prolly-cli".to_string());

    let tree_root = db.get_root_hash();
    let commit = repo.commit(
        &tree_root,
        &format!("Import from {}", sqlite_path.display()),
        None,
        None,
        None,
    );

    let commit_hash = commit.compute_hash();
    println!("Commit: {}", hex::encode(&commit_hash[..8]));
    println!("Branch: main");

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
