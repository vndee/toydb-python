import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from toydb.advanced import BloomFilter, ConflictError, DatabaseError, LSMTree, SSTable


def test_bloom_no_false_negatives():
    bloom = BloomFilter(300)
    keys = [f"key:{n}" for n in range(300)] + ["", "日本語", "😀"]
    for key in keys:
        bloom.add(key)
    assert all(key in bloom for key in keys)
    assert any(f"absent:{n}" not in bloom for n in range(300))


def test_sparse_index_across_block_boundaries(tmp_path):
    records = [dict(key=f"k{i:03}", value=i, sequence=1, deleted=False) for i in range(100)]
    table = SSTable.create(tmp_path / "table", records)
    assert len(table.sparse_keys) == 4
    for record in records:
        assert table.get(record["key"]) == record
    assert table.get("k031x") is None
    with patch.object(type(table.path), "open", side_effect=AssertionError("unnecessary I/O")):
        assert table.get("z") is None


def test_versions_tombstones_null_unicode_and_reopen(tmp_path):
    with LSMTree(tmp_path, memtable_size=2, l0_limit=2, level_ratio=2) as db:
        for index in range(30):
            db.set(f"k{index:02}", index)
            db.set("shared", index)
            if index % 3 == 0:
                db.delete("shared")
            db.flush()
        db.set("日本語", 1)
        db.set("😀", 2)
        db.set("nullable", None)
        db.delete("k00")
        db.delete("shared")
        db.flush()
        assert any(level >= 2 and tables for level, tables in db.levels.items())
    with LSMTree(tmp_path) as db:
        missing = object()
        assert db.get("shared", missing) is missing
        assert db.get("k00", missing) is missing
        assert db.get("nullable", missing) is None
        assert db.get("日本語") == 1
        assert db.get("😀") == 2
        rows = list(db.range_query("", "\U0010ffff"))
        assert len(rows) == 32
        assert rows == sorted(rows)
        assert len(dict(rows)) == len(rows)


def test_unflushed_commit_recovers_after_process_exit(tmp_path):
    script = '''
import os, sys
from toydb.advanced import LSMTree
db = LSMTree(sys.argv[1])
db.set("old", 1)
with db.transaction() as tx:
    tx.delete("old")
    tx.set("alice", 90)
    tx.set("bob", 110)
os._exit(0)
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    with LSMTree(tmp_path) as db:
        assert db.get("old") is None
        assert db.get("alice") == 90
        assert db.get("bob") == 110


def test_torn_wal_tail_removed_before_new_write(tmp_path):
    with LSMTree(tmp_path) as db:
        db.set("kept", 1)
    with (tmp_path / "wal.jsonl").open("ab") as stream:
        stream.write(b'{"batch":{"sequence":2')
    with LSMTree(tmp_path) as db:
        assert db.get("kept") == 1
        db.set("new", 2)
        db.failed = True  # Skip checkpoint to inspect WAL recovery next open.
    with LSMTree(tmp_path) as db:
        assert db.get("new") == 2


def test_corrupt_complete_wal_is_not_silently_discarded(tmp_path):
    with LSMTree(tmp_path):
        pass
    (tmp_path / "wal.jsonl").write_bytes(b'{"bad":"record"}\n')
    with pytest.raises(DatabaseError, match="Corrupt complete"):
        LSMTree(tmp_path)


def test_transaction_abort_conflict_and_atomic_wal(tmp_path):
    with LSMTree(tmp_path) as db:
        db.set("a", 100)
        db.set("b", 100)
        with pytest.raises(RuntimeError):
            with db.transaction() as tx:
                tx.set("a", 0)
                raise RuntimeError("abort")
        assert db.get("a") == 100
        first, second = db.transaction(), db.transaction()
        first.set("a", 90)
        first.set("b", 110)
        first.commit()
        second.set("a", 80)
        with pytest.raises(ConflictError):
            second.commit()
        batch = json.loads((tmp_path / "wal.jsonl").read_text().splitlines()[-1])["batch"]
        assert len(batch["operations"]) == 2
        assert db.get("a") + db.get("b") == 200


def test_mutable_values_do_not_bypass_wal(tmp_path):
    with LSMTree(tmp_path) as db:
        original = {"items": [1]}
        db.set("a", original)
        original["items"].append(2)
        returned = db.get("a")
        returned["items"].append(3)
        assert db.get("a") == {"items": [1]}
        with db.transaction() as tx:
            returned = tx.get("a")
            returned["items"].append(4)
        assert db.get("a") == {"items": [1]}


def test_parallel_transactions_retry_without_lost_updates(tmp_path):
    with LSMTree(tmp_path, memtable_size=5) as db:
        db.set("counter", 0)
        def worker():
            for _ in range(30):
                while True:
                    try:
                        with db.transaction() as tx:
                            tx.set("counter", tx.get("counter") + 1)
                        break
                    except ConflictError:
                        pass
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: worker(), range(4)))
        assert db.get("counter") == 120


def test_manifest_failure_preserves_old_tables_and_wal(tmp_path):
    db = LSMTree(tmp_path)
    db.set("key", "before")
    db.flush()
    db.set("key", "after")
    with patch("toydb.advanced._atomic_write", wraps=__import__("toydb.advanced", fromlist=["x"])._atomic_write) as write:
        original = write._mock_wraps
        def fail_manifest(path, data):
            if path.name == "MANIFEST.json":
                raise OSError("injected manifest failure")
            return original(path, data)
        write.side_effect = fail_manifest
        with pytest.raises(OSError):
            db.flush()
    db.close()
    with LSMTree(tmp_path) as reopened:
        assert reopened.get("key") == "after"


def test_directory_owner_and_closed_operations(tmp_path):
    db = LSMTree(tmp_path)
    with pytest.raises(BlockingIOError):
        LSMTree(tmp_path)
    db.close()
    with pytest.raises(DatabaseError, match="closed"):
        db.set("a", 1)
    with LSMTree(tmp_path) as reopened:
        reopened.set("a", 1)


def test_reject_part1_directory_and_non_json_value(tmp_path):
    (tmp_path / "data.db").write_bytes(b"legacy")
    with pytest.raises(DatabaseError, match="Part 1"):
        LSMTree(tmp_path)
    (tmp_path / "data.db").unlink()
    with LSMTree(tmp_path) as db:
        with pytest.raises(ValueError):
            db.set(1, "invalid key")
        with pytest.raises(ValueError):
            db.set("nan", float("nan"))
        with pytest.raises(TypeError):
            db.set("set", {1, 2})


def test_randomized_history_matches_dictionary_across_reopens(tmp_path):
    import random
    rng = random.Random(7)
    expected = {}
    for _ in range(8):
        with LSMTree(tmp_path, memtable_size=3, l0_limit=2, level_ratio=2) as db:
            for _ in range(30):
                key = rng.choice(["", "a", "b", "c", "日本語", "😀"])
                if rng.randrange(3) == 0:
                    db.delete(key)
                    expected.pop(key, None)
                else:
                    value = rng.choice([None, 0, 1, ["nested"], {"x": 5}])
                    db.set(key, value)
                    expected[key] = value
                if rng.randrange(4) == 0:
                    db.flush()
                assert list(db.range_query("", "\U0010ffff")) == sorted(expected.items())


def test_wal_reset_failure_after_manifest_is_recoverable(tmp_path):
    import toydb.advanced as advanced
    db = LSMTree(tmp_path)
    with db.transaction() as tx:
        tx.set("a", 1)
        tx.set("b", 2)
    original = advanced._atomic_write
    def fail_reset(path, data):
        if path.name == "wal.jsonl":
            raise OSError("injected reset failure")
        return original(path, data)
    with patch("toydb.advanced._atomic_write", side_effect=fail_reset):
        with pytest.raises(OSError):
            db.flush()
    db.close()
    with LSMTree(tmp_path) as reopened:
        assert list(reopened.range_query("a", "b")) == [("a", 1), ("b", 2)]
        assert reopened.sequence == 1


def test_truncated_transaction_has_no_partial_effect(tmp_path):
    db = LSMTree(tmp_path)
    db.set("a", 100)
    db.flush()
    with db.transaction() as tx:
        tx.set("a", 90)
        tx.set("b", 10)
    db.failed = True  # Preserve the transaction WAL without a checkpoint.
    db.close()
    wal = tmp_path / "wal.jsonl"
    complete = wal.read_bytes()
    wal.write_bytes(complete[:len(complete) // 2])
    with LSMTree(tmp_path) as reopened:
        assert reopened.get("a") == 100
        assert reopened.get("b") is None
