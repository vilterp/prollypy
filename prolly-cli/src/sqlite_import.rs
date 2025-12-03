use prolly_core::{Repo, DB};
use rusqlite::Connection;
use std::path::Path;
use std::time::Instant;

/// Import a SQLite database into a prolly repository.
///
/// Takes an already-opened `Repo` and imports all tables from the SQLite database
/// at `sqlite_path`.
pub fn import_sqlite(
    repo: &Repo,
    repo_path_display: &Path,
    sqlite_path: &Path,
    batch_size: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    println!("Importing SQLite database: {}", sqlite_path.display());
    println!("Target repository: {}", repo_path_display.display());
    println!("Batch size: {}", batch_size);
    println!();

    // Open SQLite database
    let sqlite_conn = Connection::open(sqlite_path)?;

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
