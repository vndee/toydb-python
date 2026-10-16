"""An educational, single-process LSM store with explicit durability boundaries.

JSON values, POSIX file locking, coarse thread synchronization; not production storage.
The original Part 1 implementation remains in toydb.database for comparison.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import threading
import uuid
from pathlib import Path

import fcntl

from .database import DatabaseError


class ConflictError(DatabaseError):
    """A transaction's snapshot is older than the latest committed write."""


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _copy(value):
    return json.loads(_encode(value))


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path, data):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class BloomFilter:
    """Stable double-hashed Bloom filter; a positive answer is only a hint."""
    def __init__(self, capacity, false_positive_rate=0.01):
        if capacity < 1 or not 0 < false_positive_rate < 1:
            raise ValueError("positive capacity and probability in (0, 1) required")
        self.bit_count = max(8, math.ceil(-capacity * math.log(false_positive_rate)
                                        / math.log(2) ** 2))
        self.hash_count = max(1, round(self.bit_count / capacity * math.log(2)))
        self.bits = bytearray((self.bit_count + 7) // 8)

    def _positions(self, key):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        first = int.from_bytes(digest[:16], "big")
        step = int.from_bytes(digest[16:], "big") | 1
        for index in range(self.hash_count):
            yield (first + index * step) % self.bit_count

    def add(self, key):
        for position in self._positions(key):
            self.bits[position // 8] |= 1 << (position % 8)

    def __contains__(self, key):
        return all(self.bits[p // 8] & (1 << (p % 8)) for p in self._positions(key))


class SSTable:
    """Immutable JSON-line records with a rebuilt Bloom filter and sparse index."""
    block_size = 32

    def __init__(self, path):
        self.path = Path(path)
        self.sparse_keys, self.offsets = [], []
        keys = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = json.loads(line)
                key = record["key"]
                if keys and key <= keys[-1]:
                    raise DatabaseError("SSTable keys are not strictly sorted")
                if len(keys) % self.block_size == 0:
                    self.sparse_keys.append(key)
                    self.offsets.append(offset)
                keys.append(key)
        self.count = len(keys)
        self.bloom = BloomFilter(max(1, self.count))
        for key in keys:
            self.bloom.add(key)
        self.minimum = keys[0] if keys else None
        self.maximum = keys[-1] if keys else None

    @classmethod
    def create(cls, path, records):
        _atomic_write(Path(path), b"".join(_encode(r) + b"\n" for r in records))
        return cls(path)

    def get(self, key):
        if not self.count or key < self.minimum or key > self.maximum:
            return None
        if key not in self.bloom:
            return None
        block = max(0, bisect.bisect_right(self.sparse_keys, key) - 1)
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[block])
            for _ in range(self.block_size):
                line = stream.readline()
                if not line:
                    break
                record = json.loads(line)
                if record["key"] == key:
                    return record
                if record["key"] > key:
                    break
        return None

    def records(self):
        with self.path.open("rb") as stream:
            for line in stream:
                yield json.loads(line)


class LSMTree:
    """Small teaching engine: one directory owner, multiple serialized threads.

    Each commit is one checksummed, fsynced WAL line. A durable manifest defines
    the live SSTable set. Level 0 may overlap; each higher level has one table.
    """
    def __init__(self, base_path, memtable_size=1000, l0_limit=4, level_ratio=10):
        if memtable_size < 1 or l0_limit < 2 or level_ratio < 2:
            raise ValueError("invalid size/level limits")
        self.path = Path(base_path)
        self.path.mkdir(parents=True, exist_ok=True)
        if (any((self.path / name).exists() for name in ("data.db", "wal.log"))
                or list(self.path.glob("sstable_*.db"))):
            raise DatabaseError("Use a new directory; Part 1 files need explicit migration")
        self.lock = threading.RLock()
        self.closed = False
        self.failed = False
        self.memtable_size, self.l0_limit = memtable_size, l0_limit
        self.level_ratio = level_ratio
        self.memtable = {}
        self.levels = {}
        self.sequence = 0
        self.checkpoint_sequence = 0
        self._owner = (self.path / "LOCK").open("a+b")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load()
        except BaseException:
            self._owner.close()
            raise

    def _check(self):
        if self.closed:
            raise DatabaseError("Database is closed")
        if self.failed:
            raise DatabaseError("I/O failed; close and reopen before continuing")

    @staticmethod
    def _key(key):
        if not isinstance(key, str):
            raise ValueError("Key must be a string")

    def _load(self):
        manifest_path = self.path / "MANIFEST.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_bytes())
            if manifest["format"] != 1:
                raise DatabaseError("Unsupported manifest format")
            self.checkpoint_sequence = self.sequence = manifest["checkpoint_sequence"]
            for item in manifest["tables"]:
                filename = item["file"]
                if Path(filename).name != filename:
                    raise DatabaseError("Invalid table name in manifest")
                self.levels.setdefault(item["level"], []).append(SSTable(self.path / filename))
        wal_path = self.path / "wal.jsonl"
        if not wal_path.exists():
            _atomic_write(wal_path, b"")
        data = wal_path.read_bytes()
        complete_end = data.rfind(b"\n") + 1
        previous = 0
        for line in data[:complete_end].splitlines():
            try:
                envelope = json.loads(line)
                batch = envelope["batch"]
                checksum = hashlib.sha256(_encode(batch)).hexdigest()
                if checksum != envelope["sha256"]:
                    raise ValueError("WAL checksum mismatch")
                seq = batch["sequence"]
                if seq <= previous:
                    raise ValueError("WAL sequence is not increasing")
                previous = seq
                if seq > self.checkpoint_sequence:
                    for operation in batch["operations"]:
                        self.memtable[operation["key"]] = dict(operation, sequence=seq)
                    self.sequence = max(self.sequence, seq)
            except (ValueError, KeyError, TypeError) as error:
                raise DatabaseError("Corrupt complete WAL record") from error
        # A missing newline marks an uncommitted/torn tail. Remove it before append.
        if complete_end != len(data):
            _atomic_write(wal_path, data[:complete_end])

    def _append(self, batch):
        envelope = {"batch": batch, "sha256": hashlib.sha256(_encode(batch)).hexdigest()}
        try:
            with (self.path / "wal.jsonl").open("ab") as stream:
                stream.write(_encode(envelope) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self.failed = True
            raise

    def _install(self, levels, checkpoint_sequence):
        manifest = {"format": 1, "checkpoint_sequence": checkpoint_sequence,
                    "tables": [{"file": t.path.name, "level": level}
                               for level, tables in sorted(levels.items()) for t in tables]}
        try:
            _atomic_write(self.path / "MANIFEST.json", _encode(manifest) + b"\n")
        except OSError:
            self.failed = True
            raise
        self.levels = levels
        self.checkpoint_sequence = checkpoint_sequence

    def _new_table(self, records):
        return SSTable.create(self.path / ("table-" + uuid.uuid4().hex + ".jsonl"), records)

    def _flush(self):
        if not self.memtable:
            return
        table = self._new_table(self.memtable[k] for k in sorted(self.memtable))
        levels = {level: list(tables) for level, tables in self.levels.items()}
        levels.setdefault(0, []).append(table)
        self._install(levels, self.sequence)
        self.memtable = {}
        try:
            _atomic_write(self.path / "wal.jsonl", b"")
        except OSError:
            self.failed = True
            raise
        self._compact()

    @staticmethod
    def _merge(tables):
        newest = {}
        for table in tables:
            for record in table.records():
                key = record["key"]
                if key not in newest or record["sequence"] > newest[key]["sequence"]:
                    newest[key] = record
        return [newest[k] for k in sorted(newest)]

    def _compact(self):
        level = 0
        while True:
            tables = self.levels.get(level, [])
            if level == 0:
                overflow = len(tables) >= self.l0_limit
            else:
                overflow = sum(t.count for t in tables) > self.memtable_size * self.level_ratio ** level
            if not overflow:
                break
            inputs = tables + self.levels.get(level + 1, [])
            output = self._new_table(self._merge(inputs))
            levels = {n: list(ts) for n, ts in self.levels.items()}
            levels[level] = []
            levels[level + 1] = [output]
            self._install(levels, self.checkpoint_sequence)
            for table in inputs:
                try:
                    table.path.unlink()
                except OSError:
                    pass  # Unreferenced files are harmless; manifest is authoritative.
            level += 1

    def _commit(self, operations):
        self._check()
        if not operations:
            return
        if len(self.memtable) >= self.memtable_size:
            self._flush()  # Maintenance occurs before acknowledging this new commit.
        sequence = self.sequence + 1
        batch = {"sequence": sequence, "operations": operations}
        self._append(batch)
        for operation in operations:
            self.memtable[operation["key"]] = dict(operation, sequence=sequence)
        self.sequence = sequence

    def set(self, key, value):
        self._key(key)
        value = _copy(value)
        with self.lock:
            self._commit([{"key": key, "value": value, "deleted": False}])

    def delete(self, key):
        self._key(key)
        with self.lock:
            self._commit([{"key": key, "value": None, "deleted": True}])

    def get(self, key, default=None):
        self._key(key)
        with self.lock:
            self._check()
            record = self.memtable.get(key)
            for tables in self.levels.values():
                for table in tables:
                    candidate = table.get(key)
                    if candidate is not None and (record is None or candidate["sequence"] > record["sequence"]):
                        record = candidate
            if record is None or record["deleted"]:
                return default
            return _copy(record["value"])

    def _snapshot(self):
        records = self._merge(t for tables in self.levels.values() for t in tables)
        newest = {r["key"]: r for r in records}
        newest.update(self.memtable)
        return {key: _copy(r["value"]) for key, r in newest.items() if not r["deleted"]}

    def range_query(self, start_key, end_key):
        self._key(start_key)
        self._key(end_key)
        with self.lock:
            self._check()
            state = self._snapshot()
            return iter((k, state[k]) for k in sorted(state) if start_key <= k <= end_key)

    def transaction(self):
        with self.lock:
            self._check()
            return Transaction(self, self.sequence, self._snapshot())

    def flush(self):
        with self.lock:
            self._check()
            self._flush()

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                if not self.failed:
                    self._flush()
            finally:
                self.closed = True
                self._owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Transaction:
    """Whole-database optimistic validation; not MVCC or per-key concurrency."""
    def __init__(self, database, sequence, snapshot):
        self.database, self.sequence = database, sequence
        self.snapshot = snapshot
        self.writes = {}
        self.done = False

    def _check(self):
        if self.done:
            raise DatabaseError("Transaction is finished")

    def get(self, key, default=None):
        self._check()
        self.database._key(key)
        return _copy(self.snapshot[key]) if key in self.snapshot else default

    def set(self, key, value):
        self._check()
        self.database._key(key)
        value = _copy(value)
        self.snapshot[key] = value
        self.writes[key] = {"key": key, "value": value, "deleted": False}

    def delete(self, key):
        self._check()
        self.database._key(key)
        self.snapshot.pop(key, None)
        self.writes[key] = {"key": key, "value": None, "deleted": True}

    def commit(self):
        self._check()
        with self.database.lock:
            self.database._check()
            self.done = True
            if self.sequence != self.database.sequence:
                raise ConflictError("Database changed; retry the whole transaction")
            self.database._commit(list(self.writes.values()))

    def rollback(self):
        self._check()
        self.done = True
        self.writes.clear()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.done:
            self.rollback() if exc_type else self.commit()
