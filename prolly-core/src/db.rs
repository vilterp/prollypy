//! Database abstraction layer for ProllyTree.
//!
//! Provides a high-level database interface on top of the ProllyTree key-value store.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::store::BlockStore;
use crate::tree::ProllyTree;

/// Represents a table schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub columns: Vec<String>,
    pub types: Vec<String>,
    pub primary_key: Vec<String>,
}

impl Table {
    /// Create a new table schema
    pub fn new(
        name: String,
        columns: Vec<String>,
        types: Vec<String>,
        primary_key: Vec<String>,
    ) -> Self {
        Table {
            name,
            columns,
            types,
            primary_key,
        }
    }
}

/// Database abstraction layer for ProllyTree.
///
/// Stores:
/// - Table schemas at /s/<table_name>
/// - Table data at /d/<table_name>/<primary_key>
pub struct DB {
    tree: ProllyTree,
}

impl DB {
    /// Initialize database with a BlockStore instance.
    ///
    /// # Arguments
    ///
    /// * `store` - Storage backend instance
    /// * `pattern` - ProllyTree split pattern
    /// * `seed` - Random seed for rolling hash
    /// * `validate` - If true, validate tree structure during operations (slower)
    pub fn new(store: Arc<dyn BlockStore>, pattern: f64, seed: u32, validate: bool) -> Self {
        let tree = ProllyTree::new(pattern, seed, Some(store), validate);
        DB { tree }
    }

    /// Create a new table schema.
    ///
    /// # Arguments
    ///
    /// * `name` - Table name
    /// * `columns` - List of column names
    /// * `types` - List of column types
    /// * `primary_key` - List of primary key column names
    ///
    /// # Returns
    ///
    /// Created Table instance
    pub fn create_table(
        &mut self,
        name: String,
        columns: Vec<String>,
        types: Vec<String>,
        primary_key: Vec<String>,
    ) -> Table {
        let table = Table::new(name.clone(), columns, types, primary_key);

        // Store schema
        let schema_key = format!("/s/{}", name).into_bytes();
        let schema_value = serde_json::to_string(&table)
            .expect("Failed to serialize table")
            .into_bytes();

        self.tree.insert_batch(vec![(schema_key, schema_value)], false);

        table
    }

    /// Get table schema by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Table name
    ///
    /// # Returns
    ///
    /// Table instance or None if not found
    pub fn get_table(&self, name: &str) -> Option<Table> {
        let schema_key = format!("/s/{}", name).into_bytes();
        let items = self.tree.items();

        for (key, value) in items {
            if key == schema_key {
                return serde_json::from_slice(&value).ok();
            }
        }
        None
    }

    /// List all table names.
    ///
    /// # Returns
    ///
    /// List of table names
    pub fn list_tables(&self) -> Vec<String> {
        let prefix = b"/s/";
        let items = self.tree.items();
        let mut tables = Vec::new();

        for (key, _) in items {
            if key.starts_with(prefix) {
                if let Ok(table_name) = String::from_utf8(key[3..].to_vec()) {
                    tables.push(table_name);
                }
            }
        }

        tables
    }

    /// Insert rows into a table.
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the table
    /// * `rows` - Vector of row tuples (values in column order as JSON values)
    /// * `verbose` - Whether to print verbose output
    ///
    /// # Returns
    ///
    /// Number of rows inserted
    pub fn insert_rows(
        &mut self,
        table_name: &str,
        rows: Vec<Vec<serde_json::Value>>,
        verbose: bool,
    ) -> usize {
        let table = self
            .get_table(table_name)
            .expect(&format!("Table {} does not exist", table_name));

        let mut mutations = Vec::new();

        for row in &rows {
            // Build primary key from row values
            let pk_value = if table.primary_key == vec!["rowid"] {
                // Special case: first element is the rowid
                row[0].to_string()
            } else {
                let pk_indices: Vec<_> = table
                    .primary_key
                    .iter()
                    .map(|col| table.columns.iter().position(|c| c == col).unwrap())
                    .collect();
                let pk_parts: Vec<_> = pk_indices
                    .iter()
                    .map(|&i| row[i].to_string().trim_matches('"').to_string())
                    .collect();
                pk_parts.join("/")
            };

            // Create key-value pair
            let key = format!("/d/{}/{}", table_name, pk_value).into_bytes();
            let value = serde_json::to_string(row)
                .expect("Failed to serialize row")
                .into_bytes();

            mutations.push((key, value));
        }

        // Sort mutations by key for better performance
        mutations.sort_by(|a, b| a.0.cmp(&b.0));

        self.tree.insert_batch(mutations, verbose);
        rows.len()
    }

    /// Get the root hash of the underlying tree
    pub fn get_root_hash(&self) -> crate::Hash {
        self.tree.get_root_hash()
    }

    /// Get reference to the underlying tree
    pub fn get_tree(&self) -> &ProllyTree {
        &self.tree
    }

    /// Get reference to the block store
    pub fn get_store(&self) -> &Arc<dyn BlockStore> {
        self.tree.store()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;

    #[test]
    fn test_create_table() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut db = DB::new(store, 0.01, 42, false);

        let table = db.create_table(
            "users".to_string(),
            vec!["id".to_string(), "name".to_string()],
            vec!["INTEGER".to_string(), "TEXT".to_string()],
            vec!["id".to_string()],
        );

        assert_eq!(table.name, "users");
        assert_eq!(table.columns.len(), 2);
    }

    #[test]
    fn test_get_table() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut db = DB::new(store, 0.01, 42, false);

        db.create_table(
            "users".to_string(),
            vec!["id".to_string(), "name".to_string()],
            vec!["INTEGER".to_string(), "TEXT".to_string()],
            vec!["id".to_string()],
        );

        let table = db.get_table("users");
        assert!(table.is_some());
        assert_eq!(table.unwrap().name, "users");
    }

    #[test]
    fn test_list_tables() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut db = DB::new(store, 0.01, 42, false);

        db.create_table(
            "users".to_string(),
            vec!["id".to_string()],
            vec!["INTEGER".to_string()],
            vec!["id".to_string()],
        );

        db.create_table(
            "posts".to_string(),
            vec!["id".to_string()],
            vec!["INTEGER".to_string()],
            vec!["id".to_string()],
        );

        let tables = db.list_tables();
        assert_eq!(tables.len(), 2);
        assert!(tables.contains(&"users".to_string()));
        assert!(tables.contains(&"posts".to_string()));
    }

    #[test]
    fn test_insert_rows() {
        let store = Arc::new(MemoryBlockStore::new());
        let mut db = DB::new(store, 0.01, 42, false);

        db.create_table(
            "users".to_string(),
            vec!["id".to_string(), "name".to_string()],
            vec!["INTEGER".to_string(), "TEXT".to_string()],
            vec!["id".to_string()],
        );

        let rows = vec![
            vec![
                serde_json::Value::Number(1.into()),
                serde_json::Value::String("Alice".to_string()),
            ],
            vec![
                serde_json::Value::Number(2.into()),
                serde_json::Value::String("Bob".to_string()),
            ],
        ];

        let count = db.insert_rows("users", rows, false);
        assert_eq!(count, 2);
    }
}
