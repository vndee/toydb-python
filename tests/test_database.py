import os
import shutil
import pytest
from threading import Thread
from time import sleep
from toydb.database import LSMTree, DatabaseError


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


def test_memtable_flush(db_path):
    db = LSMTree(db_path)

    # Add enough entries to trigger flush
    for i in range(1100):
        db.set(f"key_{i}", f"value_{i}")

    # Verify data after flush
    assert db.get("key_0") == "value_0"
    assert db.get("key_1099") == "value_1099"


def test_delete_operations(db_path):
    db = LSMTree(db_path)

    # Test delete existing key
    db.set("key1", "value1")
    assert db.get("key1") == "value1"
    db.delete("key1")
    assert db.get("key1") is None

    # Test delete non-existent key
    db.delete("nonexistent_key")  # Should not raise error

    # Test set after delete
    db.set("key1", "new_value")
    assert db.get("key1") == "new_value"


def test_range_query(db_path):
    db = LSMTree(db_path)

    # Insert test data
    test_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    for k, v in test_data.items():
        db.set(k, v)

    # Test range query
    results = list(db.range_query("b", "d"))
    assert len(results) == 3
    assert results == [("b", 2), ("c", 3), ("d", 4)]

    # Test empty range
    results = list(db.range_query("x", "z"))
    assert len(results) == 0


def test_compaction(db_path):
    db = LSMTree(db_path)

    # Create enough data to trigger multiple flushes and compaction
    for i in range(2000):
        db.set(f"key_{i}", f"value_{i}")

    # Overwrite some values to create obsolete entries
    for i in range(1000):
        db.set(f"key_{i}", f"updated_value_{i}")

    # Verify data after compaction
    assert db.get("key_0") == "updated_value_0"
    assert db.get("key_999") == "updated_value_999"
    assert db.get("key_1000") == "value_1000"


def test_concurrent_operations(db_path):
    db = LSMTree(db_path)

    def writer_thread():
        for i in range(100):
            db.set(f"thread_key_{i}", f"thread_value_{i}")
            sleep(0.001)  # Small delay to increase chance of concurrency issues

    # Create multiple writer threads
    threads = [Thread(target=writer_thread) for _ in range(5)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify all data was written correctly
    for i in range(100):
        assert db.get(f"thread_key_{i}") == f"thread_value_{i}"


def test_error_handling(db_path):
    db = LSMTree(db_path)

    # Test invalid key type
    with pytest.raises(ValueError):
        db.set(123, "value")  # Non-string key

    with pytest.raises(ValueError):
        db.get(456)  # Non-string key

    # Test operation on invalid path - use a file path instead of directory
    test_file = os.path.join(db_path, "test_file.txt")
    # Create a file where we want the directory
    with open(test_file, "w") as f:
        f.write("test")

    # Now try to create a database where a file exists (should raise DatabaseError)
    with pytest.raises(DatabaseError):
        LSMTree(test_file)

    # Clean up
    os.remove(test_file)
    db.close()


def test_edge_cases(db_path):
    db = LSMTree(db_path)

    # Test empty string key
    db.set("", "empty_key")
    assert db.get("") == "empty_key"

    # Test various value types
    test_values = [
        None,
        42,
        3.14,
        ["list", "of", "items"],
        {"key": "value"},
        True,
        "",
    ]

    for i, value in enumerate(test_values):
        db.set(f"type_test_{i}", value)
        assert db.get(f"type_test_{i}") == value


def test_recovery(db_path):
    """Test database recovery with optimized operations"""
    # Create a database instance
    db1 = LSMTree(db_path)

    # Add some data to the database
    db1.set("key1", "value1")
    db1.set("key2", "value2")
    assert db1.get("key1") == "value1"
    assert db1.get("key2") == "value2"

    # Close the database to force a flush
    db1.close()

    # Create a new instance of the database
    db2 = LSMTree(db_path)

    # Verify data is still present after recovery
    value1 = db2.get("key1")
    value2 = db2.get("key2")

    assert value1 == "value1"
    assert value2 == "value2"

    db2.close()


if __name__ == "__main__":
    pytest.main([__file__])
