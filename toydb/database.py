import os
import json
import pickle
import bisect
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Iterator
from pathlib import Path
from threading import RLock


class DatabaseError(Exception):
    """Base class for database exceptions"""

    pass


class WALEntry:
    def __init__(self, operation: str, key: str, value: Any):
        self.timestamp = datetime.utcnow().isoformat()
        self.operation = operation  # 'set' or 'delete'
        self.key = key
        self.value = value

    def serialize(self) -> str:
        """Convert the entry to a string for storage"""
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "operation": self.operation,
                "key": self.key,
                "value": self.value,
            }
        )


class WALStore:
    def __init__(self, data_file: str, wal_file: str):
        self.data_file = data_file
        self.wal_file = wal_file
        self.data: Dict[str, Any] = {}
        self._recover()

    def _append_wal(self, entry: WALEntry):
        """Write an entry to the log file"""
        try:
            with open(self.wal_file, "a") as f:
                f.write(entry.serialize() + "\n")
                f.flush()  # Ensure it's written to disk
                os.fsync(f.fileno())  # Force disk write
        except IOError as e:
            raise DatabaseError(f"Failed to write to WAL: {e}")

    def _recover(self):
        """Rebuild database state from WAL if needed"""
        try:
            # First load the last checkpoint
            if os.path.exists(self.data_file):
                with open(self.data_file, "rb") as f:
                    self.data = pickle.load(f)

            # Then replay any additional changes from WAL
            if os.path.exists(self.wal_file):
                with open(self.wal_file, "r") as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            entry = json.loads(line)
                            if entry["operation"] == "set":
                                self.data[entry["key"]] = entry["value"]
                            elif entry["operation"] == "delete":
                                self.data.pop(entry["key"], None)
        except (IOError, json.JSONDecodeError, pickle.PickleError) as e:
            raise DatabaseError(f"Recovery failed: {e}")

    def set(self, key: str, value: Any):
        """Set a key-value pair with WAL"""
        entry = WALEntry("set", key, value)
        self._append_wal(entry)
        self.data[key] = value

    def delete(self, key: str):
        """Delete a key with WAL"""
        entry = WALEntry("delete", key, None)
        self._append_wal(entry)
        self.data.pop(key, None)

    def checkpoint(self):
        """Create a checkpoint of current state"""
        temp_file = f"{self.data_file}.tmp"
        try:
            # Write to temporary file first
            with open(temp_file, "wb") as f:
                pickle.dump(self.data, f)
                f.flush()
                os.fsync(f.fileno())

            # Atomically replace old file
            shutil.move(temp_file, self.data_file)

            # Clear WAL - just truncate instead of opening in 'w' mode
            if os.path.exists(self.wal_file):
                with open(self.wal_file, "r+") as f:
                    f.truncate(0)
                    f.flush()
                    os.fsync(f.fileno())
        except IOError as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise DatabaseError(f"Checkpoint failed: {e}")


class MemTable:
    def __init__(self, max_size: int = 1000):
        self.entries: List[Tuple[str, Any]] = []
        self.max_size = max_size

    def add(self, key: str, value: Any):
        """Add or update a key-value pair"""
        idx = bisect.bisect_left([k for k, _ in self.entries], key)
        if idx < len(self.entries) and self.entries[idx][0] == key:
            self.entries[idx] = (key, value)
        else:
            self.entries.insert(idx, (key, value))

    def get(self, key: str) -> Optional[Any]:
        """Get value for key"""
        idx = bisect.bisect_left([k for k, _ in self.entries], key)
        if idx < len(self.entries) and self.entries[idx][0] == key:
            return self.entries[idx][1]
        return None

    def is_full(self) -> bool:
        """Check if memtable has reached max size"""
        return len(self.entries) >= self.max_size

    def range_scan(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Any]]:
        """Scan entries within key range"""
        start_idx = bisect.bisect_left([k for k, _ in self.entries], start_key)
        end_idx = bisect.bisect_right([k for k, _ in self.entries], end_key)
        return iter(self.entries[start_idx:end_idx])


class SSTable:
    def __init__(self, filename: str):
        self.filename = filename
        self.index: Dict[str, int] = {}

        if os.path.exists(filename):
            self._load_index()

    def _load_index(self):
        """Load index from existing SSTable file"""
        try:
            with open(self.filename, "rb") as f:
                # Read index position from start of file
                f.seek(0)
                index_pos = int.from_bytes(f.read(8), "big")

                # Read index from end of file
                f.seek(index_pos)
                self.index = pickle.load(f)
        except (IOError, pickle.PickleError) as e:
            raise DatabaseError(f"Failed to load SSTable index: {e}")

    def write_memtable(self, memtable: MemTable):
        """Save memtable to disk as SSTable"""
        temp_file = f"{self.filename}.tmp"
        try:
            with open(temp_file, "wb") as f:
                # Write index size for recovery
                index_pos = f.tell()
                f.write(b"\0" * 8)  # Placeholder for index position

                # Write data
                for key, value in memtable.entries:
                    offset = f.tell()
                    self.index[key] = offset
                    entry = pickle.dumps((key, value))
                    f.write(len(entry).to_bytes(4, "big"))
                    f.write(entry)

                # Write index at end
                index_offset = f.tell()
                pickle.dump(self.index, f)

                # Update index position at start of file
                f.seek(index_pos)
                f.write(index_offset.to_bytes(8, "big"))

                f.flush()
                os.fsync(f.fileno())

            # Atomically rename temp file
            shutil.move(temp_file, self.filename)
        except IOError as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise DatabaseError(f"Failed to write SSTable: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value for key from SSTable"""
        if key not in self.index:
            return None

        try:
            with open(self.filename, "rb") as f:
                f.seek(self.index[key])
                size = int.from_bytes(f.read(4), "big")
                entry = pickle.loads(f.read(size))
                return entry[1]
        except (IOError, pickle.PickleError) as e:
            raise DatabaseError(f"Failed to read from SSTable: {e}")

    def range_scan(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Any]]:
        """Scan entries within key range"""
        keys = sorted(k for k in self.index.keys() if start_key <= k <= end_key)
        for key in keys:
            value = self.get(key)
            if value is not None:
                yield (key, value)


class LSMTree:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        try:
            # Check if path exists and is a file
            if self.base_path.exists() and self.base_path.is_file():
                raise DatabaseError(f"Cannot create database: '{base_path}' is a file")

            self.base_path.mkdir(parents=True, exist_ok=True)

        except (OSError, FileExistsError) as e:
            raise DatabaseError(
                f"Failed to initialize database at '{base_path}': {str(e)}"
            )

        self.memtable = MemTable()
        self.sstables: List[SSTable] = []
        self.max_sstables = 5  # Limit on number of SSTables
        self.lock = RLock()

        self.wal = WALStore(
            str(self.base_path / "data.db"), str(self.base_path / "wal.log")
        )

        self._load_sstables()
        if len(self.sstables) > self.max_sstables:
            self._compact()

    def _load_sstables(self):
        """Load existing SSTables from disk"""
        self.sstables.clear()

        for file in sorted(self.base_path.glob("sstable_*.db")):
            self.sstables.append(SSTable(str(file)))

    def set(self, key: str, value: Any):
        """Set a key-value pair"""
        with self.lock:
            if not isinstance(key, str):
                raise ValueError("Key must be a string")

            # 1. Safety first: Write to WAL
            self.wal.set(key, value)

            # 2. Write to memory table (fast!)
            self.memtable.add(key, value)

            # 3. If memory table is full, save to disk
            if self.memtable.is_full():
                self._flush_memtable()

    def _flush_memtable(self):
        """Flush memtable to disk as new SSTable"""
        if not self.memtable.entries:
            return  # Skip if empty

        # Create new SSTable with a unique name
        sstable = SSTable(str(self.base_path / f"sstable_{len(self.sstables)}.db"))
        sstable.write_memtable(self.memtable)
        # Add to our list of SSTables
        self.sstables.append(sstable)

        # Create fresh memory table
        self.memtable = MemTable()

        # Create a checkpoint in WAL
        self.wal.checkpoint()

        # Compact if we have too many SSTables
        if len(self.sstables) > self.max_sstables:
            self._compact()

    def _compact(self):
        """Merge multiple SSTables into one"""
        try:
            # Create merged memtable
            merged = MemTable(max_size=float("inf"))

            # Merge all SSTables
            for sstable in self.sstables:
                for key, value in sstable.range_scan("", "~"):  # Full range
                    merged.add(key, value)

            # Write merged data to new SSTable
            new_sstable = SSTable(str(self.base_path / "sstable_compacted.db"))
            new_sstable.write_memtable(merged)

            # Remove old SSTables
            old_files = [sst.filename for sst in self.sstables]
            self.sstables = [new_sstable]

            # Delete old files
            for file in old_files:
                try:
                    os.remove(file)
                except OSError:
                    pass  # Ignore deletion errors

        except Exception as e:
            raise DatabaseError(f"Compaction failed: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value for key"""
        with self.lock:
            if not isinstance(key, str):
                raise ValueError("Key must be a string")

            # 1. Check memtable first (newest data)
            value = self.memtable.get(key)
            if value is not None:
                return value

            # 2. Check each SSTable, newest to oldest
            for sstable in reversed(self.sstables):
                value = sstable.get(key)
                if value is not None:
                    return value

            # 3. Not found anywhere
            return None

    def delete(self, key: str):
        """Delete a key"""
        with self.lock:
            self.wal.delete(key)
            self.set(key, None)  # Use None as tombstone

    def range_query(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Any]]:
        """Perform a range query"""
        with self.lock:
            # Get from memtable
            for key, value in self.memtable.range_scan(start_key, end_key):
                yield (key, value)

            # Get from each SSTable
            seen_keys = set()
            for sstable in reversed(self.sstables):
                for key, value in sstable.range_scan(start_key, end_key):
                    if key not in seen_keys:
                        seen_keys.add(key)
                        if value is not None:  # Skip tombstones
                            yield (key, value)

    def close(self):
        """Ensure all data is persisted to disk"""
        with self.lock:
            if self.memtable.entries:  # If there's data in memtable
                self._flush_memtable()

            self.wal.checkpoint()  # Ensure WAL is up-to-date


def main():
    # Example usage
    db = LSMTree("./mydb")

    try:
        # Store some user data
        db.set("user:1", {"name": "Alice", "email": "alice@example.com", "age": 30})

        # Read it back
        user = db.get("user:1")
        print(f"User: {user}")

        # Store many items
        for i in range(1000):
            db.set(f"item:{i}", {"name": f"Item {i}", "price": random.randint(1, 100)})

        # Range query example
        print("\nItems 10-15:")
        for key, value in db.range_query("item:10", "item:15"):
            print(f"{key}: {value}")

    except DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    import random

    main()
