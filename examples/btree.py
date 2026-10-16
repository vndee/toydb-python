"""In-memory B-tree insertion/search example; no persistence or deletion."""
from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass
class Node:
    leaf: bool = True
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    children: list = field(default_factory=list)


class BTree:
    def __init__(self, minimum_degree=3):
        if not isinstance(minimum_degree, int) or minimum_degree < 2:
            raise ValueError("minimum_degree must be an integer >= 2")
        self.degree = minimum_degree
        self.root = Node()

    def get(self, key, default=None):
        node = self.root
        while True:
            index = bisect_left(node.keys, key)
            if index < len(node.keys) and node.keys[index] == key:
                return node.values[index]
            if node.leaf:
                return default
            node = node.children[index]

    def _split(self, parent, index):
        degree = self.degree
        left = parent.children[index]
        right = Node(leaf=left.leaf)
        parent.keys.insert(index, left.keys[degree - 1])
        parent.values.insert(index, left.values[degree - 1])
        right.keys, left.keys = left.keys[degree:], left.keys[:degree - 1]
        right.values, left.values = left.values[degree:], left.values[:degree - 1]
        if not left.leaf:
            right.children, left.children = left.children[degree:], left.children[:degree]
        parent.children.insert(index + 1, right)

    def set(self, key, value):
        if len(self.root.keys) == 2 * self.degree - 1:
            self.root = Node(leaf=False, children=[self.root])
            self._split(self.root, 0)
        self._insert(self.root, key, value)

    def _insert(self, node, key, value):
        index = bisect_left(node.keys, key)
        if index < len(node.keys) and node.keys[index] == key:
            node.values[index] = value
            return
        if node.leaf:
            node.keys.insert(index, key)
            node.values.insert(index, value)
            return
        if len(node.children[index].keys) == 2 * self.degree - 1:
            self._split(node, index)
            if key == node.keys[index]:
                node.values[index] = value
                return
            if key > node.keys[index]:
                index += 1
        self._insert(node.children[index], key, value)

    def items(self):
        def walk(node):
            for index, key in enumerate(node.keys):
                if not node.leaf:
                    yield from walk(node.children[index])
                yield key, node.values[index]
            if not node.leaf:
                yield from walk(node.children[-1])
        return walk(self.root)
