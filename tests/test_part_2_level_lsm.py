import os
import shutil
import pytest
from threading import Thread
from time import sleep
from misc.part_2_level_lsm import LSMTree


@pytest.fixture
def db_path():
    path = "./test_db"
    yield path
    # Cleanup after tests
    if os.path.exists(path):
        shutil.rmtree(path)


def test_basic_operations(db_path):
    db = LSMTree(db_path)

    # Test single key-value
    db.set("test_key", "test_value")
    assert db.get("test_key") == "test_value"

    # Test overwrite
    db.set("test_key", "new_value")
    assert db.get("test_key") == "new_value"

    # Test non-existent key
    assert db.get("missing_key") is None

    db.close()


def test_memtable_flush_to_level0(db_path):
    db = LSMTree(db_path)

    # Add enough entries to trigger multiple flushes to Level 0
    for i in range(1100):
        db.set(f"key_{i}", f"value_{i}")

    # Verify data is still accessible
    assert db.get("key_0") == "value_0"
    assert db.get("key_1099") == "value_1099"

    # Verify Level 0 has files
    assert len(db.levels[0].sstables) > 0

    db.close()


def test_level_compaction(db_path):
    db = LSMTree(db_path)

    # Add enough data to trigger multiple flushes and compactions
    # We'll write in batches to force multiple L0 SSTables
    batch_size = 250  # Less than memtable size to control flushes
    total_entries = 5000

    for batch in range(0, total_entries, batch_size):
        # Write a batch of entries
        for i in range(batch, min(batch + batch_size, total_entries)):
            db.set(f"key_{i}", f"value_{i}")

        # Force flush after each batch
        db._flush_memtable()

        # Print current state for debugging
        print(f"After batch {batch}:")
        print(f"L0 files: {len(db.levels[0].sstables)}")
        print(f"L1 files: {len(db.levels[1].sstables)}")

    # Additional writes to ensure compaction
    for i in range(total_entries, total_entries + 1000):
        db.set(f"key_{i}", f"value_{i}")

    # Final flush to ensure all data is on disk
    db._flush_memtable()

    # Verify that files were compacted into higher levels
    print("Final state:")
    print(f"L0 files: {len(db.levels[0].sstables)}")
    print(f"L1 files: {len(db.levels[1].sstables)}")

    assert len(db.levels[1].sstables) > 0, "No files in Level 1 after compaction"

    # Verify data integrity across levels
    for i in range(0, total_entries, 500):
        value = db.get(f"key_{i}")
        assert value == f"value_{i}", f"Wrong value for key_{i}: got {value}"

    db.close()


def test_level_size_limits(db_path):
    """Test that level size limits are respected and trigger appropriate compactions"""
    db = LSMTree(db_path)

    # Helper function to log level states
    def print_level_states():
        print("\nCurrent level states:")
        for level in db.levels:
            total_size = level.size()
            file_count = len(level.sstables)
            print(f"Level {level.level_num}:")
            print(f"  Files: {file_count}")
            print(f"  Total size: {total_size / 1024:.2f} KB")
            print(f"  Max size: {level.max_size / 1024:.2f} KB")

    # Create data entries of known size
    # Each entry will be approximately 100 bytes
    value_template = "x" * 90  # Base value size
    batch_size = 100  # Number of entries per batch
    total_batches = 50  # Total number of batches to write

    print("\nStarting data insertion...")

    # Insert data in batches to trigger multiple compactions
    for batch in range(total_batches):
        print(f"\nWriting batch {batch + 1}/{total_batches}")

        # Write batch of entries
        for i in range(batch_size):
            key = f"key_{batch}_{i}"
            # Make value size vary slightly to avoid compression
            value = f"{value_template}_{batch}_{i}"
            db.set(key, value)

        # Force flush after each batch
        db._flush_memtable()

        # Log current state
        print_level_states()

        # Verify level constraints are maintained
        for level in db.levels:
            # Check level size constraints
            assert (
                level.size() <= level.max_size
            ), f"Level {level.level_num} exceeded max size"

            # Check file count constraints
            if level.max_files is not None:
                assert (
                    len(level.sstables) <= level.max_files
                ), f"Level {level.level_num} exceeded max files"

        # Verify data can still be retrieved
        # Check a sample of keys from current batch
        for i in range(0, batch_size, 10):
            key = f"key_{batch}_{i}"
            expected_value = f"{value_template}_{batch}_{i}"
            actual_value = db.get(key)
            assert (
                actual_value == expected_value
            ), f"Data verification failed for key: {key}"

    print("\nFinal state after all insertions:")
    print_level_states()

    # Final verification
    print("\nPerforming final verifications...")

    # 1. Verify size limits
    for level in db.levels:
        assert (
            level.size() <= level.max_size
        ), f"Level {level.level_num} final size exceeds limit"

    # 2. Verify file count limits
    for level in db.levels:
        if level.max_files is not None:
            assert (
                len(level.sstables) <= level.max_files
            ), f"Level {level.level_num} final file count exceeds limit"

    # 3. Verify data integrity with range query
    print("\nVerifying data integrity with range queries...")
    all_data = list(
        db.range_query("key_0_0", f"key_{total_batches - 1}_{batch_size - 1}")
    )
    assert len(all_data) > 0, "No data found in range query"

    # 4. Verify random sampling of data
    print("\nVerifying random samples of data...")
    import random

    sample_size = min(100, total_batches * batch_size)
    for _ in range(sample_size):
        batch = random.randint(0, total_batches - 1)
        i = random.randint(0, batch_size - 1)
        key = f"key_{batch}_{i}"
        expected_value = f"{value_template}_{batch}_{i}"
        actual_value = db.get(key)
        assert (
            actual_value == expected_value
        ), f"Random sampling verification failed for key: {key}"

    print("\nTest completed successfully!")
    db.close()


def test_range_query_across_levels(db_path):
    db = LSMTree(db_path)

    # Insert data that will span multiple levels
    test_ranges = {
        "a": list(range(100)),
        "b": list(range(100, 200)),
        "c": list(range(200, 300)),
    }

    # Insert data in reverse order to ensure different levels
    for prefix in reversed(list(test_ranges.keys())):
        for num in test_ranges[prefix]:
            db.set(f"{prefix}_{num}", f"value_{num}")
        # Force flush to create new SSTable
        db._flush_memtable()

    # Test range query spanning multiple levels
    results = list(db.range_query("b_100", "b_150"))
    assert len(results) == 51
    assert all(k.startswith("b_") for k, _ in results)

    # Test range query with overwrites
    for i in range(100, 150):
        db.set(f"b_{i}", f"new_value_{i}")

    results = list(db.range_query("b_100", "b_150"))
    assert len(results) == 51
    assert all(v.startswith("new_value_") for _, v in results)

    db.close()


def test_concurrent_operations_with_compaction(db_path):
    db = LSMTree(db_path)

    def writer_thread(start_range, end_range):
        for i in range(start_range, end_range):
            db.set(f"thread_key_{i}", f"thread_value_{i}")
            if i % 100 == 0:  # Occasional sleep to allow compaction
                sleep(0.001)

    # Create multiple writer threads with different ranges
    threads = [
        Thread(target=writer_thread, args=(i * 1000, (i + 1) * 1000)) for i in range(5)
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify data integrity
    for i in range(5000):
        assert db.get(f"thread_key_{i}") == f"thread_value_{i}"

    db.close()


def test_delete_with_compaction(db_path):
    db = LSMTree(db_path)

    # Insert data
    for i in range(1000):
        db.set(f"key_{i}", f"value_{i}")

    # Delete some keys
    for i in range(0, 1000, 2):
        db.delete(f"key_{i}")

    # Force compaction
    db._flush_memtable()

    # Verify deletes are persistent
    for i in range(1000):
        if i % 2 == 0:
            assert db.get(f"key_{i}") is None
        else:
            assert db.get(f"key_{i}") == f"value_{i}"

    # Insert new values for deleted keys
    for i in range(0, 1000, 2):
        db.set(f"key_{i}", f"new_value_{i}")

    # Verify new values
    for i in range(0, 1000, 2):
        assert db.get(f"key_{i}") == f"new_value_{i}"

    db.close()


def test_recovery_with_levels(db_path):
    # First database instance
    db1 = LSMTree(db_path)

    # Add data that will be spread across levels
    for i in range(3000):
        db1.set(f"key_{i}", f"value_{i}")
        if i % 1000 == 0:
            db1._flush_memtable()  # Force some flushes

    # Verify initial state
    assert len(db1.levels[0].sstables) > 0

    # Close first instance
    db1.close()

    # Create new instance
    db2 = LSMTree(db_path)

    # Verify data is preserved across levels
    for i in range(3000):
        assert db2.get(f"key_{i}") == f"value_{i}"

    # Verify level structure is maintained
    assert len(db2.levels[0].sstables) > 0

    db2.close()


def test_edge_cases_with_levels(db_path):
    db = LSMTree(db_path)

    # Test empty string keys at different levels
    db.set("", "level0")
    db._flush_memtable()
    db.set("", "level1")
    db._flush_memtable()
    assert db.get("") == "level1"  # Should get most recent value

    # Test very large values
    large_value = "x" * 1024 * 1024  # 1MB value
    db.set("large_key", large_value)
    assert db.get("large_key") == large_value

    # Test keys that should span levels
    for i in range(100):
        key = "z" * i  # Keys with increasing lengths
        db.set(key, f"value_{i}")
        if i % 10 == 0:
            db._flush_memtable()

    # Verify all values are retrievable
    for i in range(100):
        key = "z" * i
        assert db.get(key) == f"value_{i}"

    db.close()


if __name__ == "__main__":
    pytest.main([__file__])
