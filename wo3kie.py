from utils import cnll

from heapq import *
from math import log, pi, sin, cos, exp
import random

import copy
import numpy as np
import os

from pyquil.api import QVMConnection
from pyquil.quil import Program

# private keys

from wo3kie_private import API_KEY, USER_ID

# config

PYQUIL_CONFIG = f"""
[Rigetti Forest]
url: https://api.rigetti.com/qvm
key: {API_KEY}
user_id: {USER_ID}
"""

np.set_printoptions(linewidth=9999)
np.set_printoptions(precision=2)

#

# tree utils

class INode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.get_depth() + 1

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_depth(self):
        return self.depth

class Node(INode):
    def __init__(self, parent=None, qubits=None, gate=None, cost=None):
        super().__init__(parent)

        self.qubits = qubits
        self.gate = gate
        self.cost = cost

    def get_qubits(self):
        return self.qubits

    def get_gate(self):
        return self.gate

    def get_cost(self):
        return cost

# heap utils

def make_heap():
    return []

def push_heap(heap, node, cost):
    heappush(heap, (cost, node))

def pop_heap(heap):
    return heappop(heap)

def empty_heap(heap):
    return len(heap) == 0

# Qubit

class QuBits:
    def __init__(self, size=1):
        self.size = size
        self.data = np.zeros(2 ** size)
        self.data[0] = 1

    def get_size(self):
        return self.size

    def get_amplitudes(self):
        return self.data

    def get_probabilities(self):
        return np.abs(self.get_amplitudes()) ** 2

    def copy(self):
        qubits = QuBits()
        qubits.size = self.size
        qubits.data = self.data.copy()
        return qubits

#

class I:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        qubits = qubits.copy()

        return qubits

    def dump(self):
        return "I {}".format(self.n)

class RZ:
    def __init__(self, n, phi):
        self.n = n
        self.phi = phi

    def run(self, qubits):
        rz = np.array(
            [[cos(self.phi / 2) - 1j * sin(self.phi / 2), 0],
            [0, cos(self.phi / 2) + 1j * sin(self.phi / 2)]]
        )

        i = np.eye(2)

        if self.n == 0:
            gate = rz
        else:
            gate = i

        for n in range(1, qubits.get_size()):
            if n == self.n:
                gate = np.kron(gate, rz)
            else:
                gate = np.kron(gate, i)

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)

        return result

    def dump(self):
        return "RZ ({:.2f}) {}".format(self.phi, self.n)

class RX:
    def __init__(self, n, phi):
        self.n = n
        self.phi = phi

    def run(self, qubits):
        rx = np.array(
            [[cos(self.phi / 2), -1j * sin(self.phi / 2)],
            [-1j * sin(self.phi / 2), cos(self.phi / 2)]]
        )

        i = np.eye(2)

        if self.n == 0:
            gate = rx
        else:
            gate = i

        for n in range(1, qubits.get_size()):
            if n == self.n:
                gate = np.kron(gate, rx)
            else:
                gate = np.kron(gate, i)

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)

        return result

    def dump(self):
        return "RX ({:.2f}) {}".format(self.phi, self.n)

class CN:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        cn = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]]
        )

        n = 0
        i = np.eye(2)

        if self.n == 0:
            gate = cn
            n += 1
        else:
            gate = i

        n += 1

        while n < qubits.get_size():
            if n == self.n:
                gate = np.kron(gate, cn)
                n += 1
            else:
                gate = np.kron(gate, i)

            n += 1

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)
        return result

    def dump(self):
        return "CN {} {}".format(self.n, self.n + 1)

class Swap:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        swap = np.array(
            [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]]
        )

        n = 0
        i = np.eye(2)

        if self.n == 0:
            gate = swap
            n += 1
        else:
            gate = i

        n += 1

        while n < qubits.get_size():
            if n == self.n:
                gate = np.kron(gate, swap)
                n += 1
            else:
                gate = np.kron(gate, i)

            n += 1

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)
        return result

    def dump(self):
        return "SWAP {} {}".format(self.n, self.n + 1)


class ICN:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        icn = np.array(
            [[0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        )

        n = 0
        i = np.eye(2)

        if self.n == 0:
            gate = icn
            n += 1
        else:
            gate = i

        n += 1

        while n < qubits.get_size():
            if n == self.n:
                gate = np.kron(gate, icn)
                n += 1
            else:
                gate = np.kron(gate, i)

            n += 1

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)
        return result

    def dump(self):
        return "CN {} {}".format(self.n + 1, self.n)

class R:
    def __init__(self, n, phi):
        self.n = n
        self.phi = phi

    def run(self, qubits):
        r = np.array(
            [[1, 0],
            [0, np.exp(1j * self.phi)]]
        )

        i = np.eye(2)

        if self.n == 0:
            gate = r
        else:
            gate = i

        for n in range(1, qubits.get_size()):
            if n == self.n:
                gate = np.kron(gate, r)
            else:
                gate = np.kron(gate, i)

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)

        return result

    def dump(self):
        return "R {}".format(self.n)

class X:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        x = np.array(
            [[0, 1],
            [1, 0]]
        )

        i = np.eye(2)

        if self.n == 0:
            gate = x
        else:
            gate = i

        for n in range(1, qubits.get_size()):
            if n == self.n:
                gate = np.kron(gate, x)
            else:
                gate = np.kron(gate, i)

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)

        return result

    def dump(self):
        return "X {}".format(self.n)

class H:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        h = np.array(
            [[0, 1],
            [1, 0]]
        )

        i = np.eye(2)

        if self.n == 0:
            gate = h
        else:
            gate = i

        for n in range(1, qubits.get_size()):
            if n == self.n:
                gate = np.kron(gate, h)
            else:
                gate = np.kron(gate, i)

        result = QuBits(qubits.get_size())
        result.data = np.dot(gate, qubits.data)

        return result

    def dump(self):
        return "H {}".format(self.n)

#

with open(os.path.expanduser('~/.pyquil_config'), 'w') as f:
    f.write(PYQUIL_CONFIG)

qvm = QVMConnection()
pro = Program()

#

def find_lcz(actual, expected):
    n_qubits = int(np.log2(len(actual)))
    cost = cnll(actual, expected)

    heap = make_heap()
    push_heap(heap, Node(parent=None, qubits=QuBits(4), gate=I(0), cost=cost), (cost, random.random()))

    operations = []

    #for qubit_id in range(0, n_qubits):
    #    operations.append(X(qubit_id))

    for qubit_id in range(0, n_qubits):
        operations.append(H(qubit_id))

    for qubit_id in range(0, n_qubits - 1):
        operations.append(CN(qubit_id))

    #for qubit_id in range(0, n_qubits - 1):
    #    operations.append(Swap(qubit_id))

    for qubit_id in range(0, n_qubits):
        for angle in [0.927295]:
            operations.append(RZ(qubit_id, angle))
            operations.append(RX(qubit_id, angle))
            #operations.append(R(qubit_id, angle))

    counter = 0
    min_cost = 9999
    best_solution = None
    visited = set()

    while (counter < 500000) and (empty_heap(heap) is False):
        ((cost, rnd), node) = pop_heap(heap)
        qubits = node.qubits

        for operation in operations:
            #if node.get_depth() >= 120:
            #    continue

            counter += 1    
            
            if counter % 1000 == 0:
                print(counter, end='\r')      

            new_qubits = operation.run(qubits)
            
            if new_qubits in visited:
                continue
            else:
                visited.add(new_qubits)
            
            new_cost = cnll(new_qubits.get_probabilities(), expected)

            #print(qubits.get_probabilities())
            #print(operation.dump())
            #print(new_qubits.get_probabilities())

            if new_cost < 0:
                continue

            if (new_cost > 1.2 * cost):
                continue

            if new_cost < min_cost:
                min_cost = new_cost
                best_solution = node
                print(cost, " -> ", new_cost)

            #print()
            push_heap(heap, Node(parent=node, qubits=new_qubits, gate=operation, cost=cost), (cost, random.random()))

    node = best_solution
    nodes = []
    
    while node is not None:
        nodes.append(node)
        node = node.get_parent()

    for node in nodes[::-1]:
        print(node.get_gate().dump())

    print(best_solution.get_qubits().get_probabilities())
#

actual = np.array([1] + 15 * [0])

expected = np.array([
    1/6, # 0000
    0/6, # 0001
    0/6, # 0010
    1/6, # 0011
    0/6, # 0100
    1/6, # 0101
    0/6, # 0110
    0/6, # 0111
    0/6, # 1000
    0/6, # 1001
    1/6, # 1010
    0/6, # 1011
    0/6, # 1100
    1/6, # 1101
    0/6, # 1110
    1/6, # 1111
])

# main

def main():
    random.seed(0)
    find_lcz(actual, expected)

main()

qubits = QuBits(3)
print(qubits.get_amplitudes())

qubits = CN(1).run(qubits)
print(qubits.get_amplitudes())
