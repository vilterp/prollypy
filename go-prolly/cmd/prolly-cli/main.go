package main

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	prolly "github.com/vilterp/go-prolly/prolly-core"
	_ "github.com/mattn/go-sqlite3"
	"github.com/spf13/cobra"
)

const (
	DefaultCacheSize = 10000
	DefaultFanout    = 32
)

var (
	repoPath  string
	cacheSize int
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "prolly",
		Short: "Prolly tree CLI",
	}

	rootCmd.PersistentFlags().StringVar(&repoPath, "repo", ".prolly", "Repository path")
	rootCmd.PersistentFlags().IntVar(&cacheSize, "cache-size", DefaultCacheSize, "LRU cache size")

	rootCmd.AddCommand(initCmd())
	rootCmd.AddCommand(importCmd())
	rootCmd.AddCommand(gcCmd())
	rootCmd.AddCommand(branchCmd())
	rootCmd.AddCommand(checkoutCmd())
	rootCmd.AddCommand(logCmd())
	rootCmd.AddCommand(getCmd())
	rootCmd.AddCommand(setCmd())
	rootCmd.AddCommand(dumpCmd())
	rootCmd.AddCommand(diffCmd())
	rootCmd.AddCommand(printTreeCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func openRepo() (*prolly.Repo, error) {
	blocksPath := filepath.Join(repoPath, "blocks")
	commitsPath := filepath.Join(repoPath, "commits.db")

	blockStore, err := prolly.NewCachedFSBlockStore(blocksPath, cacheSize)
	if err != nil {
		return nil, err
	}

	commitStore, err := prolly.NewSqliteCommitGraphStore(commitsPath)
	if err != nil {
		return nil, err
	}

	return prolly.NewRepo(blockStore, commitStore, "go-prolly"), nil
}

// init command
func initCmd() *cobra.Command {
	var author string

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Initialize a new repository",
		RunE: func(cmd *cobra.Command, args []string) error {
			blocksPath := filepath.Join(repoPath, "blocks")
			commitsPath := filepath.Join(repoPath, "commits.db")

			if err := os.MkdirAll(blocksPath, 0755); err != nil {
				return err
			}

			blockStore, err := prolly.NewCachedFSBlockStore(blocksPath, cacheSize)
			if err != nil {
				return err
			}

			commitStore, err := prolly.NewSqliteCommitGraphStore(commitsPath)
			if err != nil {
				return err
			}

			repo := prolly.NewRepo(blockStore, commitStore, author)
			commitHash := repo.InitEmpty()

			fmt.Printf("Initialized repository at %s\n", repoPath)
			fmt.Printf("Initial commit: %s\n", commitHash.Hex())
			return nil
		},
	}

	cmd.Flags().StringVar(&author, "author", "go-prolly", "Default author")
	return cmd
}

// import command
func importCmd() *cobra.Command {
	var batchSize int

	cmd := &cobra.Command{
		Use:   "import <sqlite-path>",
		Short: "Import SQLite database",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			sqlitePath := args[0]

			repo, err := openRepo()
			if err != nil {
				return err
			}

			// Open source SQLite
			srcDB, err := sql.Open("sqlite3", sqlitePath)
			if err != nil {
				return err
			}
			defer srcDB.Close()

			// Get current tree
			treeRoot, commit, err := repo.GetTreeAtRef("HEAD")
			if err != nil {
				return err
			}

			root := repo.BlockStore.GetNode(treeRoot)
			tree := prolly.NewProllyTreeFromRoot(repo.BlockStore, root, commit.Pattern, commit.Seed)
			db := prolly.NewDB(tree)

			// Get tables
			rows, err := srcDB.Query(`SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`)
			if err != nil {
				return err
			}

			var tables []string
			for rows.Next() {
				var name string
				rows.Scan(&name)
				tables = append(tables, name)
			}
			rows.Close()

			totalRows := 0
			start := time.Now()

			for _, tableName := range tables {
				// Get schema
				schemaRows, err := srcDB.Query(fmt.Sprintf("PRAGMA table_info(%s)", tableName))
				if err != nil {
					continue
				}

				var columns, types []string
				for schemaRows.Next() {
					var cid int
					var name, typ string
					var notnull, pk int
					var dflt interface{}
					schemaRows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk)
					columns = append(columns, name)
					types = append(types, typ)
				}
				schemaRows.Close()

				// Get primary key
				pkRows, err := srcDB.Query(fmt.Sprintf("PRAGMA table_info(%s)", tableName))
				if err != nil {
					continue
				}

				var primaryKey []string
				for pkRows.Next() {
					var cid int
					var name, typ string
					var notnull, pk int
					var dflt interface{}
					pkRows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk)
					if pk > 0 {
						primaryKey = append(primaryKey, name)
					}
				}
				pkRows.Close()

				if len(primaryKey) == 0 {
					primaryKey = []string{"rowid"}
					columns = append([]string{"rowid"}, columns...)
					types = append([]string{"INTEGER"}, types...)
				}

				// Create table
				db.CreateTable(tableName, columns, types, primaryKey)

				// Import data
				dataRows, err := srcDB.Query(fmt.Sprintf("SELECT rowid, * FROM %s", tableName))
				if err != nil {
					continue
				}

				colNames, _ := dataRows.Columns()
				var batch []map[string]interface{}

				for dataRows.Next() {
					vals := make([]interface{}, len(colNames))
					valPtrs := make([]interface{}, len(colNames))
					for i := range vals {
						valPtrs[i] = &vals[i]
					}
					dataRows.Scan(valPtrs...)

					row := make(map[string]interface{})
					for i, col := range colNames {
						row[col] = vals[i]
					}
					batch = append(batch, row)

					if batchSize > 0 && len(batch) >= batchSize {
						db.InsertRows(tableName, batch)
						totalRows += len(batch)
						batch = nil
					}
				}
				dataRows.Close()

				if len(batch) > 0 {
					db.InsertRows(tableName, batch)
					totalRows += len(batch)
				}

				fmt.Printf("Imported table %s\n", tableName)
			}

			elapsed := time.Since(start)
			rowsPerSec := float64(totalRows) / elapsed.Seconds()

			// Commit
			commitHash, err := repo.Commit(db.GetRootHash(), fmt.Sprintf("Import from %s", sqlitePath), commit.Pattern, commit.Seed)
			if err != nil {
				return err
			}

			fmt.Printf("Imported %d rows in %v (%.0f rows/sec)\n", totalRows, elapsed, rowsPerSec)
			fmt.Printf("Commit: %s\n", commitHash.Hex())
			return nil
		},
	}

	cmd.Flags().IntVar(&batchSize, "batch-size", 0, "Batch size (0 for single batch)")
	return cmd
}

// gc command
func gcCmd() *cobra.Command {
	var dryRun bool

	cmd := &cobra.Command{
		Use:   "gc",
		Short: "Garbage collect unreachable nodes",
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			roots := repo.GetReachableTreeRoots()
			stats := prolly.GarbageCollect(repo.BlockStore, roots, dryRun)

			if dryRun {
				fmt.Println("Dry run:")
			}
			fmt.Printf("Total nodes: %d\n", stats.TotalNodes)
			fmt.Printf("Reachable: %d\n", stats.ReachableNodes)
			fmt.Printf("Garbage: %d\n", stats.GarbageNodes)
			return nil
		},
	}

	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Don't actually delete")
	return cmd
}

// branch command
func branchCmd() *cobra.Command {
	var from string

	cmd := &cobra.Command{
		Use:   "branch [name]",
		Short: "Create or list branches",
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			if len(args) == 0 {
				// List branches
				branches := repo.ListBranches()
				head := repo.GetHead()

				// Sort branch names
				var names []string
				for name := range branches {
					names = append(names, name)
				}
				sort.Strings(names)

				for _, name := range names {
					hash := branches[name]
					marker := "  "
					if name == head {
						marker = "* "
					}
					fmt.Printf("%s%s %s\n", marker, name, hash.Hex()[:8])
				}
			} else {
				// Create branch
				name := args[0]
				if from == "" {
					from = "HEAD"
				}
				if err := repo.CreateBranch(name, from); err != nil {
					return err
				}
				fmt.Printf("Created branch %s\n", name)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&from, "from", "", "Create from ref")
	return cmd
}

// checkout command
func checkoutCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "checkout <branch>",
		Short: "Switch branches",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			branchName := args[0]
			if err := repo.Checkout(branchName); err != nil {
				return err
			}

			commitHash, _ := repo.ResolveRef(branchName)
			commit := repo.CommitGraphStore.GetCommit(commitHash)
			fmt.Printf("Switched to branch '%s'\n", branchName)
			fmt.Printf("Commit: %s\n", commitHash.Hex()[:8])
			if commit != nil {
				fmt.Printf("Message: %s\n", commit.Message)
			}
			return nil
		},
	}
}

// log command
func logCmd() *cobra.Command {
	var startRef string
	var maxCount int

	cmd := &cobra.Command{
		Use:   "log",
		Short: "Show commit history",
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			if startRef == "" {
				startRef = "HEAD"
			}

			commits, err := repo.Log(startRef, maxCount)
			if err != nil {
				return err
			}

			for _, entry := range commits {
				ts := time.Unix(int64(entry.Commit.Timestamp), 0)
				fmt.Printf("commit %s\n", entry.Hash.Hex())
				fmt.Printf("Author: %s\n", entry.Commit.Author)
				fmt.Printf("Date:   %s\n", ts.Format(time.RFC1123))
				fmt.Printf("\n    %s\n\n", entry.Commit.Message)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&startRef, "start", "", "Starting ref")
	cmd.Flags().IntVar(&maxCount, "max-count", 0, "Max commits")
	return cmd
}

// get command
func getCmd() *cobra.Command {
	var fromRef string

	cmd := &cobra.Command{
		Use:   "get <key>",
		Short: "Get value by key",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			if fromRef == "" {
				fromRef = "HEAD"
			}

			treeRoot, commit, err := repo.GetTreeAtRef(fromRef)
			if err != nil {
				return err
			}

			root := repo.BlockStore.GetNode(treeRoot)
			tree := prolly.NewProllyTreeFromRoot(repo.BlockStore, root, commit.Pattern, commit.Seed)

			key := args[0]
			value := tree.Get([]byte(key))
			if value == nil {
				return fmt.Errorf("key not found: %s", key)
			}

			fmt.Println(string(value))
			return nil
		},
	}

	cmd.Flags().StringVar(&fromRef, "from", "", "Read from ref")
	return cmd
}

// set command
func setCmd() *cobra.Command {
	var message string

	cmd := &cobra.Command{
		Use:   "set <key> <value>",
		Short: "Set key-value and commit",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			treeRoot, commit, err := repo.GetTreeAtRef("HEAD")
			if err != nil {
				return err
			}

			root := repo.BlockStore.GetNode(treeRoot)
			tree := prolly.NewProllyTreeFromRoot(repo.BlockStore, root, commit.Pattern, commit.Seed)

			key, value := args[0], args[1]
			tree.InsertBatch([]struct{ Key, Value []byte }{
				{[]byte(key), []byte(value)},
			})

			if message == "" {
				message = fmt.Sprintf("Set %s", key)
			}

			commitHash, err := repo.Commit(tree.GetRootHash(), message, commit.Pattern, commit.Seed)
			if err != nil {
				return err
			}

			fmt.Printf("Commit: %s\n", commitHash.Hex())
			return nil
		},
	}

	cmd.Flags().StringVar(&message, "message", "", "Commit message")
	return cmd
}

// dump command
func dumpCmd() *cobra.Command {
	var fromRef string
	var prefix string

	cmd := &cobra.Command{
		Use:   "dump",
		Short: "Dump all keys",
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			if fromRef == "" {
				fromRef = "HEAD"
			}

			treeRoot, _, err := repo.GetTreeAtRef(fromRef)
			if err != nil {
				return err
			}

			var seekTo []byte
			if prefix != "" {
				seekTo = []byte(prefix)
			}

			cursor := prolly.NewTreeCursorWithSeek(repo.BlockStore, treeRoot, seekTo)
			count := 0

			for !cursor.Done() {
				key, value := cursor.Next()
				if key == nil {
					break
				}
				keyStr := string(key)
				if prefix != "" && !strings.HasPrefix(keyStr, prefix) {
					break
				}
				fmt.Printf("%s: %s\n", keyStr, string(value))
				count++
			}

			fmt.Printf("\nTotal: %d entries\n", count)
			return nil
		},
	}

	cmd.Flags().StringVar(&fromRef, "from", "", "Read from ref")
	cmd.Flags().StringVar(&prefix, "prefix", "", "Key prefix filter")
	return cmd
}

// diff command
func diffCmd() *cobra.Command {
	var oldRef, newRef, prefix string
	var limit int

	cmd := &cobra.Command{
		Use:   "diff",
		Short: "Compare commits",
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			if oldRef == "" {
				oldRef = "HEAD~1"
			}
			if newRef == "" {
				newRef = "HEAD"
			}

			oldRoot, _, err := repo.GetTreeAtRef(oldRef)
			if err != nil {
				return err
			}
			newRoot, _, err := repo.GetTreeAtRef(newRef)
			if err != nil {
				return err
			}

			var prefixBytes []byte
			if prefix != "" {
				prefixBytes = []byte(prefix)
			}

			events, stats := prolly.DiffWithLimit(repo.BlockStore, oldRoot, newRoot, prefixBytes, limit)

			for _, event := range events {
				switch event.Type {
				case prolly.DiffAdded:
					fmt.Printf("+ %s: %s\n", string(event.Key), string(event.Value))
				case prolly.DiffDeleted:
					fmt.Printf("- %s: %s\n", string(event.Key), string(event.OldValue))
				case prolly.DiffModified:
					fmt.Printf("~ %s: %s -> %s\n", string(event.Key), string(event.OldValue), string(event.Value))
				}
			}

			fmt.Printf("\nChanges: %d\n", len(events))
			fmt.Printf("Subtrees skipped: %d\n", stats.SubtreesSkipped)
			fmt.Printf("Nodes compared: %d\n", stats.NodesCompared)
			return nil
		},
	}

	cmd.Flags().StringVar(&oldRef, "old", "", "Old ref")
	cmd.Flags().StringVar(&newRef, "new", "", "New ref")
	cmd.Flags().StringVar(&prefix, "prefix", "", "Key prefix filter")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max events")
	return cmd
}

// print-tree command
func printTreeCmd() *cobra.Command {
	var fromRef string
	var verbose bool

	cmd := &cobra.Command{
		Use:   "print-tree",
		Short: "Visualize tree structure",
		RunE: func(cmd *cobra.Command, args []string) error {
			repo, err := openRepo()
			if err != nil {
				return err
			}

			if fromRef == "" {
				fromRef = "HEAD"
			}

			treeRoot, _, err := repo.GetTreeAtRef(fromRef)
			if err != nil {
				return err
			}

			printNode(repo.BlockStore, treeRoot, "", verbose)
			return nil
		},
	}

	cmd.Flags().StringVar(&fromRef, "from", "", "Read from ref")
	cmd.Flags().BoolVar(&verbose, "verbose", false, "Show all values")
	return cmd
}

func printNode(store prolly.BlockStore, hash prolly.Hash, indent string, verbose bool) {
	node := store.GetNode(hash)
	if node == nil {
		fmt.Printf("%s<missing>\n", indent)
		return
	}

	if node.IsLeaf {
		if verbose {
			for i := range node.Keys {
				fmt.Printf("%s%s: %s\n", indent, string(node.Keys[i]), string(node.Values[i]))
			}
		} else {
			if len(node.Keys) > 0 {
				first := string(node.Keys[0])
				last := string(node.Keys[len(node.Keys)-1])
				fmt.Printf("%sLeaf[%d]: %s..%s\n", indent, len(node.Keys), first, last)
			} else {
				fmt.Printf("%sLeaf[0]: <empty>\n", indent)
			}
		}
	} else {
		fmt.Printf("%sInternal[%d children] %s\n", indent, len(node.Values), hash.Hex()[:8])
		for i, childHashBytes := range node.Values {
			childHash := prolly.HashFromBytes(childHashBytes)
			if i < len(node.Keys) {
				fmt.Printf("%s  sep: %s\n", indent, string(node.Keys[i]))
			}
			printNode(store, childHash, indent+"  ", verbose)
		}
	}
}
