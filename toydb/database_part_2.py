import os
import json
import shutil
import mmh3  # For Bloom filter hashing
from datetime import datetime
from typing import Any, Dict, List, Optional, Iterator, Tuple
from pathlib import Path
from threading import RLock
from collections import defaultdict
from toydb.database import WALEntry as BaseWALEntry
from toydb.database import WALStore, MemTable, SSTable, DatabaseError


class WALEntry(BaseWALEntry):
    """Extended WAL entry with transaction support"""

    def __init__(
        self, operation: str, key: str, value: Any, txn_id: Optional[str] = None
    ):
        super().__init__(operation, key, value)
        self.txn_id = txn_id

    def serialize(self) -> str:
        """Convert the entry to a string for storage"""
        data = {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "key": self.key,
            "value": self.value,
            "txn_id": self.txn_id,
        }
        return json.dumps(data)


class BloomFilter:
    """Probabilistic data structure for quick membership testing"""

    def __init__(self, size: int = 100000, num_hashes: int = 4):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = bytearray((size + 7) // 8)

    def _hash_functions(self, item: str) -> Iterator[int]:
        h1 = mmh3.hash(item, 0)
        h2 = mmh3.hash(item, 1)
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        for bit_pos in self._hash_functions(item):
            byte_pos = bit_pos // 8
            bit_offset = bit_pos % 8
            self.bits[byte_pos] |= 1 << bit_offset

    def might_contain(self, item: str) -> bool:
        for bit_pos in self._hash_functions(item):
            byte_pos = bit_pos // 8
            bit_offset = bit_pos % 8
            if not (self.bits[byte_pos] & (1 << bit_offset)):
                return False
        return True


class Level:
    """Represents a level in the LSM tree"""

    def __init__(self, level_num: int, max_size: int):
        self.level_num = level_num
        self.max_size = max_size
        self.sstables = []  # type: ignore[var-annotated]

    def size(self) -> int:
        return sum(os.path.getsize(sst.filename) for sst in self.sstables)

    def needs_compaction(self) -> bool:
        return self.size() > self.max_size


class Transaction:
    """Handles atomic operations"""

    def __init__(self, db: "ImprovedLSMTree"):
        self.db = db
        self.id = datetime.utcnow().isoformat()
        self.writes: Dict[str, Any] = {}  # Pending writes
        self.reads: Dict[str, Any] = {}  # Read set for conflict detection with values
        self.committed = False
        self.active = False  # Track if transaction is active

    def __enter__(self):
        """Start the transaction"""
        self.active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the transaction"""
        if exc_type is not None:
            # If there was an exception, rollback
            self.rollback()
            return False  # Re-raise the exception

        if self.active and not self.committed:
            # If transaction wasn't explicitly committed, try to commit now
            if not self.commit():
                self.rollback()
                raise DatabaseError("Transaction failed to commit")

        return True

    def set(self, key: str, value: Any):
        """Stage a write operation"""
        if not self.active:
            raise DatabaseError("Transaction not active")
        # Automatically add to read set if not already there
        if key not in self.reads:
            self.reads[key] = self.db.get(key)
        self.writes[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Read within transaction"""
        if not self.active:
            raise DatabaseError("Transaction not active")

        if key in self.writes:
            return self.writes[key]

        value = self.db.get(key)
        self.reads[key] = value  # Store the value at read time
        return value

    def commit(self) -> bool:
        """Try to commit transaction"""
        if not self.active:
            raise DatabaseError("Transaction not active")

        try:
            with self.db.memtable_lock:  # Lock for atomic commit
                # Check for conflicts
                for key, read_value in self.reads.items():
                    current_value = self.db.get(key)
                    if current_value != read_value:
                        return False  # Conflict detected

                # Write WAL entries
                for key, value in self.writes.items():
                    self.db.wal._append_wal(WALEntry("set", key, value, self.id))

                self.db.wal._append_wal(WALEntry("commit", "", None, self.id))

                # Apply changes
                for key, value in self.writes.items():
                    self.db.memtable.add(key, value)

                self.committed = True
                return True
        finally:
            self.active = False  # Mark transaction as no longer active

    def rollback(self):
        """Discard all changes"""
        if self.active and not self.committed:
            self.db.wal._append_wal(WALEntry("rollback", "", None, self.id))
        self.writes.clear()
        self.reads.clear()
        self.active = False


class MemTableWithBloom(MemTable):
    """MemTable with Bloom filter"""

    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)
        self.bloom = BloomFilter()

    def add(self, key: str, value: Any):
        """Add key-value pair with Bloom filter update"""
        super().add(key, value)
        self.bloom.add(key)


class SSTableWithBloom(SSTable):
    """SSTable with Bloom filter"""

    def __init__(self, filename: str):
        super().__init__(filename)
        self.bloom = BloomFilter()

    def write_memtable(self, memtable: MemTable):
        """Write memtable to SSTable with Bloom filter"""
        # Add all keys to Bloom filter
        for key, _ in memtable.entries:
            self.bloom.add(key)

        super().write_memtable(memtable)

    def get(self, key: str) -> Optional[Any]:
        """Get value using Bloom filter to avoid unnecessary reads"""
        if not self.bloom.might_contain(key):
            return None
        return super().get(key)


class ImprovedLSMTree:
    """LSM Tree with improved concurrency, transactions, bloom-filter and leveled storage"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Fine-grained locks
        self.memtable_lock = RLock()
        self.sstable_locks = defaultdict(RLock)  # type: ignore[var-annotated]
        self.compaction_lock = RLock()

        # Leveled structure
        self.levels = [
            Level(0, 4),  # L0: 4 files
            Level(1, 10 * 1024 * 1024),  # L1: 10MB
            Level(2, 100 * 1024 * 1024),  # L2: 100MB
            Level(3, 1024 * 1024 * 1024),  # L3: 1GB
        ]

        self.memtable = MemTableWithBloom(max_size=1000)
        self.wal = WALStore(
            str(self.base_path / "data.db"), str(self.base_path / "wal.log")
        )

        self._load_sstables()

    def _load_sstables(self):
        """Load existing SSTables from disk into appropriate levels"""
        # Clear existing SSTables from all levels
        for level in self.levels:
            level.sstables.clear()

        # Load SSTables from disk and assign to appropriate levels
        for file in sorted(self.base_path.glob("level*_sstable_*.db")):
            # Parse level number from filename
            # Filename format: "level{level_num}_sstable_{table_num}.db"
            try:
                level_num = int(file.name.split("_")[0].replace("level", ""))
                if 0 <= level_num < len(self.levels):
                    sstable = SSTableWithBloom(str(file))
                    self.levels[level_num].sstables.append(sstable)
            except (ValueError, IndexError):
                # Skip files that don't match our naming convention
                continue

        # Load any old-format SSTables into Level 0
        for file in sorted(self.base_path.glob("sstable_*.db")):
            self.levels[0].sstables.append(SSTableWithBloom(str(file)))

        # Sort SSTables within each level by creation time
        for level in self.levels:
            level.sstables.sort(key=lambda sst: os.path.getctime(sst.filename))

    def _flush_memtable(self):
        """Flush memtable to disk as new SSTable in Level 0"""
        if not self.memtable.entries:
            return  # Skip if empty

        # Create new SSTable in Level 0
        sstable = SSTableWithBloom(
            str(self.base_path / f"level0_sstable_{len(self.levels[0].sstables)}.db")
        )
        sstable.write_memtable(self.memtable)

        # Add to Level 0
        self.levels[0].sstables.append(sstable)

        # Create fresh memory table
        self.memtable = MemTableWithBloom(self.memtable.max_size)

        # Create a checkpoint in WAL
        self.wal.checkpoint()

        # Check if Level 0 needs compaction
        if self.levels[0].needs_compaction():
            self._compact()

    def transaction(self) -> Transaction:
        """Start a new transaction"""
        return Transaction(self)

    def _select_compaction_files(self, level: Level) -> List[SSTable]:
        """Select files for compaction based on size and overlap"""
        if level.level_num == 0:
            return level.sstables

        # Select files that are causing most overlap
        # This is a simple implementation - real databases use more sophisticated strategies
        return level.sstables[:2]

    def _find_overlapping_sstables(
        self, level: Level, input_files: List[SSTable]
    ) -> List[SSTable]:
        """Find SSTables in the level that overlap with input files"""
        # Simple implementation - in practice, would use file metadata to check key ranges
        return level.sstables

    def _merge_sstables(self, files1: List[SSTable], files2: List[SSTable]) -> MemTable:
        """Merge multiple SSTables together"""
        merged = MemTableWithBloom(max_size=float("inf"))  # type: ignore[arg-type]

        # Merge all files
        all_files = files1 + files2
        for sstable in all_files:
            for key, value in sstable.range_scan("", "~"):  # Full range
                merged.add(key, value)

        return merged

    def _write_sorted_runs(self, memtable: MemTable, level_num: int) -> List[SSTable]:
        """Write sorted runs to the specified level"""
        sstable = SSTableWithBloom(
            str(
                self.base_path
                / f"level{level_num}_sstable_{len(self.levels[level_num].sstables)}.db"
            )
        )
        sstable.write_memtable(memtable)
        return [sstable]

    def _update_levels_after_compaction(
        self,
        source_level: Level,
        target_level: Level,
        old_files: List[SSTable],
        new_files: List[SSTable],
    ):
        """Update level references after compaction"""
        # Remove old files
        source_level.sstables = [
            sst for sst in source_level.sstables if sst not in old_files
        ]
        target_level.sstables.extend(new_files)

        # Delete old files from disk
        for sst in old_files:
            try:
                os.remove(sst.filename)
            except OSError:
                pass

    def _compact(self):
        """Perform leveled compaction"""
        with self.compaction_lock:
            for level_num, level in enumerate(self.levels[:-1]):
                if not level.needs_compaction():
                    continue

                # Select files to compact
                input_files = self._select_compaction_files(level)

                # Find overlapping files in next level
                next_level = self.levels[level_num + 1]
                overlapping_files = self._find_overlapping_sstables(
                    next_level, input_files
                )

                # Merge files
                merged_data = self._merge_sstables(input_files, overlapping_files)

                # Write new files
                new_sstables = self._write_sorted_runs(merged_data, level_num + 1)

                # Update level references
                self._update_levels_after_compaction(
                    level, next_level, input_files + overlapping_files, new_sstables
                )

    def set(self, key: str, value: Any):
        """Set a key-value pair with improved concurrency"""
        if not isinstance(key, str):
            raise ValueError("Key must be a string")

        with self.memtable_lock:
            self.wal.set(key, value)
            self.memtable.add(key, value)

            if self.memtable.is_full():
                with self.compaction_lock:
                    self._flush_memtable()

    def get(self, key: str) -> Optional[Any]:
        """Get value for key with Bloom filter optimization"""
        if not isinstance(key, str):
            raise ValueError("Key must be a string")

        # Check memtable
        with self.memtable_lock:
            value = self.memtable.get(key)
            if value is not None:
                return value

        # Check SSTables in each level
        for level in self.levels:
            for sstable in reversed(level.sstables):
                if not sstable.bloom.might_contain(key):
                    continue

                with self.sstable_locks[sstable.filename]:
                    value = sstable.get(key)
                    if value is not None:
                        return value

        return None

    def delete(self, key: str) -> None:
        """Delete a key"""
        with self.memtable_lock:
            if not isinstance(key, str):
                raise ValueError("Key must be a string")

            self.wal.delete(key)
            self.set(key, None)  # Use None as tombstone

    def range_query(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Any]]:
        """Perform a range query"""
        with self.memtable_lock:
            # Get from memtable
            seen_keys = set()
            for key, value in self.memtable.range_scan(start_key, end_key):
                if value is not None:  # Skip tombstones
                    seen_keys.add(key)
                    yield (key, value)

            # Get from each level, starting from newest
            for level in self.levels:
                for sstable in reversed(level.sstables):
                    for key, value in sstable.range_scan(start_key, end_key):
                        if key not in seen_keys and value is not None:
                            seen_keys.add(key)
                            yield (key, value)

    def close(self) -> None:
        """Ensure all data is persisted to disk"""
        with self.memtable_lock:
            if self.memtable.entries:
                self._flush_memtable()
            self.wal.checkpoint()


def test_database():
    """Test basic database operations"""
    # Create test directory
    test_dir = "./test_db"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Create database
    db = ImprovedLSMTree(test_dir)

    # Test basic operations
    print("\nTesting basic operations...")
    db.set("test1", "value1")
    assert db.get("test1") == "value1"
    print("✓ Basic operations passed")

    # Test successful transaction
    print("\nTesting successful transaction...")
    with db.transaction() as txn:
        txn.set("test2", "value2")
        txn.set("test3", "value3")
        assert txn.commit()  # Explicit commit

    assert db.get("test2") == "value2"
    assert db.get("test3") == "value3"
    print("✓ Successful transaction passed")

    # Test transaction rollback
    print("\nTesting transaction rollback...")
    try:
        with db.transaction() as txn:
            txn.set("test4", "value4")
            raise ValueError("Simulated error")  # Should trigger rollback
    except ValueError:
        assert db.get("test4") is None  # Should not exist after rollback
    print("✓ Transaction rollback passed")

    # Test transaction conflict
    print("\nTesting transaction conflict...")
    db.set("conflict_key", "original")

    with db.transaction() as txn1:
        # Simulate concurrent modification
        db.set("conflict_key", "modified")

        # Now try to modify the value
        txn1.set("conflict_key", "txn1_value")
        assert not txn1.commit()  # Should fail due to conflict

    assert (
        db.get("conflict_key") == "modified"
    )  # Should keep the concurrent modification
    print("✓ Transaction conflict handling passed")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_database()
