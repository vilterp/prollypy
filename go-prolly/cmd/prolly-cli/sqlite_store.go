package main

import (
	"database/sql"
	"strings"

	prolly "github.com/vilterp/go-prolly/prolly-core"
	_ "github.com/mattn/go-sqlite3"
)

// SqliteCommitGraphStore stores commits in SQLite
type SqliteCommitGraphStore struct {
	db *sql.DB
}

// NewSqliteCommitGraphStore creates a new SQLite commit store
func NewSqliteCommitGraphStore(dbPath string) (*SqliteCommitGraphStore, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	// Create tables
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS commits (
			hash BLOB PRIMARY KEY,
			tree_root BLOB,
			message TEXT,
			timestamp REAL,
			author TEXT,
			pattern REAL,
			seed INTEGER
		);
		CREATE TABLE IF NOT EXISTS commit_parents (
			commit_hash BLOB,
			parent_hash BLOB,
			idx INTEGER
		);
		CREATE TABLE IF NOT EXISTS refs (
			name TEXT PRIMARY KEY,
			commit_hash BLOB
		);
		CREATE TABLE IF NOT EXISTS metadata (
			key TEXT PRIMARY KEY,
			value TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_commit_parents ON commit_parents(commit_hash);
	`)
	if err != nil {
		return nil, err
	}

	return &SqliteCommitGraphStore{db: db}, nil
}

func (s *SqliteCommitGraphStore) PutCommit(hash prolly.Hash, commit prolly.Commit) {
	_, err := s.db.Exec(
		`INSERT OR REPLACE INTO commits (hash, tree_root, message, timestamp, author, pattern, seed)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		hash[:], commit.TreeRoot[:], commit.Message, commit.Timestamp, commit.Author, commit.Pattern, commit.Seed,
	)
	if err != nil {
		return
	}

	// Delete old parents
	s.db.Exec(`DELETE FROM commit_parents WHERE commit_hash = ?`, hash[:])

	// Insert parents
	for i, parent := range commit.Parents {
		s.db.Exec(
			`INSERT INTO commit_parents (commit_hash, parent_hash, idx) VALUES (?, ?, ?)`,
			hash[:], parent[:], i,
		)
	}
}

func (s *SqliteCommitGraphStore) GetCommit(hash prolly.Hash) *prolly.Commit {
	var commit prolly.Commit
	var treeRoot []byte

	err := s.db.QueryRow(
		`SELECT tree_root, message, timestamp, author, pattern, seed FROM commits WHERE hash = ?`,
		hash[:],
	).Scan(&treeRoot, &commit.Message, &commit.Timestamp, &commit.Author, &commit.Pattern, &commit.Seed)
	if err != nil {
		return nil
	}
	commit.TreeRoot = prolly.HashFromBytes(treeRoot)

	// Get parents
	rows, err := s.db.Query(
		`SELECT parent_hash FROM commit_parents WHERE commit_hash = ? ORDER BY idx`,
		hash[:],
	)
	if err != nil {
		return nil
	}
	defer rows.Close()

	for rows.Next() {
		var parent []byte
		rows.Scan(&parent)
		commit.Parents = append(commit.Parents, prolly.HashFromBytes(parent))
	}

	return &commit
}

func (s *SqliteCommitGraphStore) GetParents(hash prolly.Hash) []prolly.Hash {
	commit := s.GetCommit(hash)
	if commit == nil {
		return nil
	}
	return commit.Parents
}

func (s *SqliteCommitGraphStore) SetRef(name string, hash prolly.Hash) {
	s.db.Exec(
		`INSERT OR REPLACE INTO refs (name, commit_hash) VALUES (?, ?)`,
		name, hash[:],
	)
}

func (s *SqliteCommitGraphStore) GetRef(name string) *prolly.Hash {
	var hashBytes []byte
	err := s.db.QueryRow(`SELECT commit_hash FROM refs WHERE name = ?`, name).Scan(&hashBytes)
	if err != nil {
		return nil
	}
	hash := prolly.HashFromBytes(hashBytes)
	return &hash
}

func (s *SqliteCommitGraphStore) ListRefs() map[string]prolly.Hash {
	result := make(map[string]prolly.Hash)
	rows, err := s.db.Query(`SELECT name, commit_hash FROM refs`)
	if err != nil {
		return result
	}
	defer rows.Close()

	for rows.Next() {
		var name string
		var hashBytes []byte
		rows.Scan(&name, &hashBytes)
		result[name] = prolly.HashFromBytes(hashBytes)
	}
	return result
}

func (s *SqliteCommitGraphStore) SetHead(refName string) {
	s.db.Exec(`INSERT OR REPLACE INTO metadata (key, value) VALUES ('head', ?)`, refName)
}

func (s *SqliteCommitGraphStore) GetHead() string {
	var head string
	s.db.QueryRow(`SELECT value FROM metadata WHERE key = 'head'`).Scan(&head)
	return head
}

func (s *SqliteCommitGraphStore) FindCommitByPrefix(prefix string) *prolly.Hash {
	rows, err := s.db.Query(`SELECT hash FROM commits`)
	if err != nil {
		return nil
	}
	defer rows.Close()

	for rows.Next() {
		var hashBytes []byte
		rows.Scan(&hashBytes)
		hash := prolly.HashFromBytes(hashBytes)
		if strings.HasPrefix(hash.Hex(), prefix) {
			return &hash
		}
	}
	return nil
}

func (s *SqliteCommitGraphStore) ListCommits() []prolly.Hash {
	var result []prolly.Hash
	rows, err := s.db.Query(`SELECT hash FROM commits`)
	if err != nil {
		return result
	}
	defer rows.Close()

	for rows.Next() {
		var hashBytes []byte
		rows.Scan(&hashBytes)
		result = append(result, prolly.HashFromBytes(hashBytes))
	}
	return result
}

func (s *SqliteCommitGraphStore) Close() error {
	return s.db.Close()
}
