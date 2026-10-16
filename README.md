## Building a Database From Scratch

- [Building a Database From Scratch: Understanding LSM Trees and Storage Engines (Part 1)](https://blog.duy.dev/building-a-database-from-scratch-understanding-lsm-trees-and-storage-engines-part-1/)

## Companion implementation for Parts 2–5

`toydb/advanced.py` adds a separate educational engine. The original Part 1 code
in `toydb/database.py` stays available for comparing the stages of the tutorial.
Use a **new directory** for the advanced format; automatic migration is not provided.

```python
from toydb.advanced import LSMTree, ConflictError

with LSMTree("./advanced-db", memtable_size=1000) as db:
    db.set("alice", 100)
    db.set("bob", 100)
    with db.transaction() as tx:
        tx.set("alice", tx.get("alice") - 10)
        tx.set("bob", tx.get("bob") + 10)
    print(list(db.range_query("alice", "bob")))
```

Implemented:

- Stable SHA-256 Bloom filters rebuilt when opening SSTables, plus a sparse index
  every 32 records. Bloom positives still require an exact lookup.
- Immutable sorted JSON-line SSTables, monotonically increasing commit sequences,
  explicit tombstones distinct from JSON `null`, and sorted deduplicated scans.
- Level 0 accumulation and geometric capacity limits for later levels. The teaching
  policy merges whole levels and keeps at most one file in each level above zero;
  it does not implement production range-partitioned compaction.
- A checksummed, fsynced WAL record per commit, including multi-key transactions.
  Recovery replays complete records and truncates an incomplete final record;
  corrupt complete records fail visibly.
- Atomic, directory-synced manifest replacement before obsolete table deletion.
- Full-copy optimistic transactions with whole-database version validation and
  read-your-writes. Conflicts raise `ConflictError`; retry the entire transaction.
- One directory owner via POSIX `flock`, and serialized operations across threads.

### Boundaries

This is a POSIX teaching implementation, not a production database. The filesystem
must support atomic same-directory rename, `fsync`, directory sync, and advisory
locking. It does not provide MVCC, per-key conflict detection, parallel commits,
background compaction, checksummed SSTables, replication, corruption repair, or
unbounded streaming scans. Transactions, compaction and range scans materialize
data in memory; tombstones are retained. A Bloom filter and sparse index are rebuilt
by scanning files at startup. Only JSON-compatible values and string keys are
supported. Values are copied at API boundaries. A transaction object belongs to one
thread. Maintenance runs before the next write when the memtable reaches its limit,
and on explicit `flush()`/`close()`. Storage failures can leave a commit outcome
uncertain; stop, close, reopen and reconcile. The tests exercise process exit and
injected failure paths, not real power-loss guarantees across filesystems.

`examples/btree.py` is a separate in-memory B-tree insertion/search illustration.
It is not a persistent B+ tree, and does not implement deletion or transactions.

### Run tests

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests cover block boundaries, Bloom membership, WAL process-exit recovery, torn
log tails, corruption rejection, manifest failures, repeated compaction, nulls,
Unicode keys, tombstones, optimistic conflicts, threaded retries, mutable values,
and B-tree ordering, occupancy and equal leaf depths. The historical Part 1 tests
also run; passing them alone should not be treated as a durability proof.
