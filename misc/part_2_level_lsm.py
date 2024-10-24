import os
from pathlib import Path
from threading import RLock
from datetime import datetime
from typing import Any, List, Optional, Tuple, Iterator
from toydb.database import DatabaseError, MemTable, SSTable, WALStore


class Level:
    """Represents a level in the LSM tree"""

    def __init__(self, level_num: int, max_size: int, max_files: Optional[int] = None):
        self.level_num = level_num
        self.max_size = max_size
        self.max_files = max_files
        self.sstables: List[SSTable] = []

    def size(self) -> int:
        """Calculate total size of all SSTables in this level"""
        return sum(os.path.getsize(sst.filename) for sst in self.sstables)

    def needs_compaction(self) -> bool:
        """Check if this level needs compaction based on size or file count"""
        if self.max_files is not None:
            return len(self.sstables) >= self.max_files
        # For other levels, trigger compaction if either:
        # 1. Total size exceeds max_size
        # 2. Number of files exceeds 10 (to prevent too many small files)
        return self.size() > self.max_size or len(self.sstables) > 10

    def get_key_range(self) -> Tuple[str, str]:
        """Get the minimum and maximum keys in this level"""
        if not self.sstables:
            return ("", "")

        all_keys = []
        for sst in self.sstables:
            if sst.index:
                all_keys.extend(sst.index.keys())

        if not all_keys:
            return ("", "")

        return (min(all_keys), max(all_keys))


class LSMTree:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.lock = RLock()

        try:
            if self.base_path.exists() and self.base_path.is_file():
                raise DatabaseError(f"Cannot create database: '{base_path}' is a file")

            self.base_path.mkdir(parents=True, exist_ok=True)

        except (OSError, FileExistsError) as e:
            raise DatabaseError(
                f"Failed to initialize database at '{base_path}': {str(e)}"
            )

        # Our "Inbox" for new data
        self.memtable = MemTable(max_size=1000)

        # Leveled storage with increasing sizes
        self.levels = [
            Level(
                level_num=0, max_size=1024 * 1024, max_files=4
            ),  # L0: 4MB, max 4 files
            Level(level_num=1, max_size=10 * 1024 * 1024),  # L1: 10MB
            Level(level_num=2, max_size=100 * 1024 * 1024),  # L2: 100MB
            Level(level_num=3, max_size=1000 * 1024 * 1024),  # L3: 1GB
        ]

        self.wal = WALStore(
            str(self.base_path / "data.db"), str(self.base_path / "wal.log")
        )

        self._load_sstables()

    def _load_sstables(self):
        """Load existing SSTables from disk"""
        # Clear existing SSTables from all levels
        for level in self.levels:
            level.sstables.clear()

        # Load SSTables from disk and assign to appropriate levels
        for file in sorted(self.base_path.glob("level*_sstable_*.db")):
            try:
                level_num = int(file.stem.split("_")[0].replace("level", ""))
                sstable = SSTable(str(file))
                self.levels[level_num].sstables.append(sstable)
            except (ValueError, IndexError):
                raise DatabaseError(f"Invalid SSTable filename format: {file}")

        # Sort SSTables within each level
        for level in self.levels[1:]:  # Skip L0
            level.sstables.sort(
                key=lambda sst: min(sst.index.keys()) if sst.index else ""
            )

    def set(self, key: str, value: Any):
        """Set a key-value pair"""
        with self.lock:
            if not isinstance(key, str):
                raise ValueError("Key must be a string")

            # Write to WAL first for durability
            self.wal.set(key, value)

            # Write to memtable
            self.memtable.add(key, value)

            # Flush if memtable is full
            if self.memtable.is_full():
                self._flush_memtable()

    def _flush_memtable(self):
        """Flush memtable to disk as new SSTable in Level 0"""
        if not self.memtable.entries:
            return

        # Create new SSTable in Level 0
        timestamp = datetime.now().timestamp()
        sstable = SSTable(str(self.base_path / f"level0_sstable_{timestamp}.db"))
        sstable.write_memtable(self.memtable)

        # Add to Level 0
        self.levels[0].sstables.append(sstable)

        # Reset memtable
        self.memtable = MemTable(max_size=self.memtable.max_size)
        self.wal.checkpoint()

        # Check if compaction is needed for L0
        if self.levels[0].needs_compaction():
            self._compact(0)
        # Also check L1 independently
        if self.levels[1].needs_compaction():
            self._compact(1)

    def _select_files_for_compaction(self, level_num: int) -> List[SSTable]:
        """Select files that should be compacted in the given level"""
        current_level = self.levels[level_num]
        next_level = (
            self.levels[level_num + 1] if level_num < len(self.levels) - 1 else None
        )

        if not next_level or not current_level.needs_compaction():
            return []

        if level_num == 0:
            # For L0, select all files when file count threshold is exceeded
            return list(current_level.sstables)

        # For other levels, select all files from current level that need compaction
        # plus overlapping files from next level
        overlapping_files = list(current_level.sstables)

        # Find overlapping files in next level
        current_min, current_max = current_level.get_key_range()
        if current_min and current_max:  # Only if we have keys in current level
            for next_sst in next_level.sstables:
                if not next_sst.index:
                    continue
                next_min = min(next_sst.index.keys())
                next_max = max(next_sst.index.keys())
                if current_min <= next_max and current_max >= next_min:
                    overlapping_files.append(next_sst)

        return overlapping_files

    def _compact(self, level_num: int):
        """Perform compaction on the specified level"""
        if level_num >= len(self.levels) - 1:
            return  # Can't compact the last level

        files_to_compact = self._select_files_for_compaction(level_num)
        if not files_to_compact:
            return

        try:
            # Create merged memtable with effectively unlimited size
            merged = MemTable(max_size=float("inf"))

            # Track all seen keys to handle overwrites and tombstones
            seen_keys = set()
            tombstones = set()

            # Process files in reverse order (newest first)
            for sstable in reversed(files_to_compact):
                for key, value in sstable.range_scan("", "~"):
                    if key not in seen_keys:
                        seen_keys.add(key)
                        if value is None:
                            # Track tombstones
                            tombstones.add(key)
                        elif key not in tombstones:
                            # Only add value if key hasn't been tombstoned
                            merged.add(key, value)

            if not merged.entries and not tombstones:  # Skip if no entries to compact
                return

            # Write to new SSTable in next level
            timestamp = datetime.now().timestamp()
            next_level = level_num + 1
            new_sstable = SSTable(
                str(self.base_path / f"level{next_level}_sstable_{timestamp}.db")
            )

            # Add tombstones to merged memtable
            for key in tombstones:
                merged.add(key, None)

            new_sstable.write_memtable(merged)

            # Update level references
            old_files = []
            for sstable in files_to_compact:
                for level in self.levels:
                    if sstable in level.sstables:
                        level.sstables.remove(sstable)
                        old_files.append(sstable.filename)
                        break

            self.levels[next_level].sstables.append(new_sstable)

            # Delete old files
            for filename in old_files:
                try:
                    os.remove(filename)
                except OSError:
                    pass  # Ignore deletion errors

            # Check if next level needs compaction
            if self.levels[next_level].needs_compaction():
                self._compact(next_level)

        except Exception as e:
            raise DatabaseError(f"Compaction failed: {str(e)}")

    def get(self, key: str) -> Optional[Any]:
        """Get value for key"""
        with self.lock:
            if not isinstance(key, str):
                raise ValueError("Key must be a string")

            # Check memtable first
            value = self.memtable.get(key)
            if value is not None:
                return value

            # Check each level, starting from L0
            for level in self.levels:
                # For L0, check all files (newest first)
                if level.level_num == 0:
                    for sstable in reversed(level.sstables):
                        value = sstable.get(key)
                        if value is not None:
                            return value
                else:
                    # For other levels, use binary search since files are sorted
                    for sstable in level.sstables:
                        if not sstable.index:
                            continue
                        min_key = min(sstable.index.keys())
                        max_key = max(sstable.index.keys())
                        if min_key <= key <= max_key:
                            value = sstable.get(key)
                            if value is not None:
                                return value

            return None

    def range_query(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Any]]:
        """Perform a range query"""
        with self.lock:
            # Dictionary to store most recent value for each key
            results = {}

            # Process L0 files in reverse order (newest to oldest)
            for sstable in reversed(self.levels[0].sstables):
                for key, value in sstable.range_scan(start_key, end_key):
                    if key not in results:
                        results[key] = value

            # Process other levels
            for level in self.levels[1:]:
                for sstable in level.sstables:
                    for key, value in sstable.range_scan(start_key, end_key):
                        if key not in results:
                            results[key] = value

            # Add memtable entries last to ensure newest values
            for key, value in self.memtable.range_scan(start_key, end_key):
                results[key] = value

            # Yield results in sorted order, skipping tombstones
            for key in sorted(results.keys()):
                if results[key] is not None:  # Skip tombstones
                    yield (key, results[key])

    def delete(self, key: str):
        """Delete a key by writing a tombstone"""
        with self.lock:
            # Write to WAL first
            self.wal.delete(key)

            # Mark as deleted in memtable
            self.memtable.add(key, None)

            # Flush if memtable is full
            if self.memtable.is_full():
                self._flush_memtable()

    def close(self):
        """Close the database, ensuring all data is persisted"""
        with self.lock:
            if self.memtable.entries:
                self._flush_memtable()
            # Force final compaction if needed
            for level_num in range(len(self.levels) - 1):
                if self.levels[level_num].needs_compaction():
                    self._compact(level_num)
            self.wal.checkpoint()
