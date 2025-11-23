package prollycore

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Table represents a database table schema
type Table struct {
	Name       string   `json:"name"`
	Columns    []string `json:"columns"`
	Types      []string `json:"types"`
	PrimaryKey []string `json:"primary_key"`
}

// DB provides a database abstraction over ProllyTree
type DB struct {
	tree *ProllyTree
}

// NewDB creates a new database
func NewDB(tree *ProllyTree) *DB {
	return &DB{tree: tree}
}

// CreateTable creates a new table
func (db *DB) CreateTable(name string, columns, types, primaryKey []string) error {
	table := Table{
		Name:       name,
		Columns:    columns,
		Types:      types,
		PrimaryKey: primaryKey,
	}

	data, err := json.Marshal(table)
	if err != nil {
		return err
	}

	key := fmt.Sprintf("/s/%s", name)
	db.tree.InsertBatch([]struct{ Key, Value []byte }{
		{[]byte(key), data},
	})

	return nil
}

// GetTable retrieves a table schema
func (db *DB) GetTable(name string) (*Table, error) {
	key := fmt.Sprintf("/s/%s", name)
	data := db.tree.Get([]byte(key))
	if data == nil {
		return nil, fmt.Errorf("table %s not found", name)
	}

	var table Table
	if err := json.Unmarshal(data, &table); err != nil {
		return nil, err
	}

	return &table, nil
}

// ListTables returns all table names
func (db *DB) ListTables() []string {
	var tables []string
	cursor := NewTreeCursorWithSeek(db.tree.Store, db.tree.GetRootHash(), []byte("/s/"))

	for !cursor.Done() {
		key, _ := cursor.Next()
		if key == nil {
			break
		}
		keyStr := string(key)
		if !strings.HasPrefix(keyStr, "/s/") {
			break
		}
		tables = append(tables, strings.TrimPrefix(keyStr, "/s/"))
	}

	return tables
}

// InsertRows inserts rows into a table
func (db *DB) InsertRows(tableName string, rows []map[string]interface{}) (BatchStats, error) {
	table, err := db.GetTable(tableName)
	if err != nil {
		return BatchStats{}, err
	}

	var mutations []struct{ Key, Value []byte }

	for _, row := range rows {
		// Build primary key
		var pkParts []string
		for _, pk := range table.PrimaryKey {
			val := row[pk]
			pkParts = append(pkParts, fmt.Sprintf("%v", val))
		}
		pk := strings.Join(pkParts, "|")
		key := fmt.Sprintf("/d/%s/%s", tableName, pk)

		// Serialize row
		data, err := json.Marshal(row)
		if err != nil {
			return BatchStats{}, err
		}

		mutations = append(mutations, struct{ Key, Value []byte }{
			[]byte(key), data,
		})
	}

	stats := db.tree.InsertBatch(mutations)
	return stats, nil
}

// GetRow retrieves a row by primary key
func (db *DB) GetRow(tableName string, pk string) (map[string]interface{}, error) {
	key := fmt.Sprintf("/d/%s/%s", tableName, pk)
	data := db.tree.Get([]byte(key))
	if data == nil {
		return nil, fmt.Errorf("row not found")
	}

	var row map[string]interface{}
	if err := json.Unmarshal(data, &row); err != nil {
		return nil, err
	}

	return row, nil
}

// GetRootHash returns the current tree root hash
func (db *DB) GetRootHash() Hash {
	return db.tree.GetRootHash()
}

// GetTree returns the underlying tree
func (db *DB) GetTree() *ProllyTree {
	return db.tree
}
