import random

import pytest

from examples.btree import BTree


@pytest.mark.parametrize("degree", [2, 3, 8])
def test_random_insert_update_and_structure(degree):
    tree = BTree(degree)
    keys = list(range(500))
    random.Random(42).shuffle(keys)
    expected = {}
    for key in keys:
        tree.set(key, str(key))
        expected[key] = str(key)
    for key in keys[::3]:
        tree.set(key, -key)
        expected[key] = -key
    assert list(tree.items()) == sorted(expected.items())
    assert all(tree.get(key) == value for key, value in expected.items())
    assert tree.get(999, "missing") == "missing"
    depths = set()
    def check(node, depth=0, low=None, high=None):
        assert node.keys == sorted(node.keys)
        assert len(node.keys) <= 2 * degree - 1
        if depth:
            assert len(node.keys) >= degree - 1
        assert len(node.values) == len(node.keys)
        assert all((low is None or low < key) and (high is None or key < high) for key in node.keys)
        if node.leaf:
            assert not node.children
            depths.add(depth)
        else:
            assert len(node.children) == len(node.keys) + 1
            bounds = [low] + node.keys + [high]
            for index, child in enumerate(node.children):
                check(child, depth + 1, bounds[index], bounds[index + 1])
    check(tree.root)
    assert len(depths) == 1
