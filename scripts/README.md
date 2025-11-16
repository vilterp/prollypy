# Table Splitting Script

This script analyzes two versions of a SQLite database to identify columns that change frequently vs rarely, then splits tables to separate stable data from operational data. The goal is to increase prolly tree commonality between database versions.

## Usage

### Mode 1: Analyze and Split Two Databases

Analyzes two databases, identifies the split specification, and applies it to both:

```bash
python scripts/split_tables.py db_a.sqlite db_b.sqlite
```

This will:
1. Analyze column change frequencies between the two databases
2. Identify tables with large column change skew
3. Generate a split specification (`split_spec.json`)
4. Apply the split to both databases, creating:
   - `db_a-split.sqlite`
   - `db_b-split.sqlite`
   - `split_spec.json`

### Mode 2: Apply Existing Split Spec

Apply a previously generated split specification to a single database:

```bash
python scripts/split_tables.py --apply-spec split_spec.json input.sqlite
```

This creates `input-split.sqlite` using the split specification.

## Options

- `--threshold PERCENT`: Columns changing less than this percentage are considered "stable" (default: 10.0)
- `--min-skew PERCENT`: Minimum difference between stable and operational columns to trigger split (default: 50.0)
- `--pattern FLOAT`: ProllyTree pattern for analysis (default: 0.01)
- `--seed INT`: ProllyTree seed for analysis (default: 42)
- `--save-spec FILE`: Save split spec to JSON file (default: split_spec.json)

## How It Works

### 1. Column Change Analysis

The script imports both databases into prolly trees and computes a diff to identify which columns changed in modified rows. For example:

```
buses: 100 modified rows
  va: 100.0%        <- operational (changes frequently)
  vm: 100.0%        <- operational
  zone: 5.0%        <- stable (rarely changes)
  name: 0.0%        <- stable
  type: 0.0%        <- stable
  owner: 0.0%       <- stable
```

### 2. Split Specification

Tables with large skew between stable and operational columns are identified for splitting. The split spec defines which columns go into each table:

```json
{
  "buses": {
    "stable": ["name", "type", "owner", "zone"],
    "operational": ["va", "vm"]
  }
}
```

### 3. Schema Transformation

The original table is split into two:

**Original:**
```
buses(rowid, name, type, owner, va, vm, zone)
```

**Split into:**
```
buses(rowid, name, type, owner, zone)              -- stable columns
buses_operational(rowid, va, vm)                   -- operational columns
```

Both tables share the same `rowid` primary key, so they can be joined:

```sql
SELECT b.name, b.type, bo.va, bo.vm
FROM buses b
JOIN buses_operational bo ON b.rowid = bo.rowid
```

## Example

Given two power grid simulation databases from different seasons:

```bash
# Analyze and split both databases
python scripts/split_tables.py BC00ALL-29WP.sqlite BC00ALL-29SP.sqlite

# Output:
#   BC00ALL-29WP-split.sqlite
#   BC00ALL-29SP-split.sqlite
#   split_spec.json
```

Now when you import these into a prolly tree repository, the stable parts (bus names, types, topology) will have high tree commonality, while only the operational parts (voltages, flows) will differ.

## Testing

Run the test suite to verify the script works correctly:

```bash
python scripts/test_split.py
```

This creates mock databases, runs the split script, and verifies the output.
