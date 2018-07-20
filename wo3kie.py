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

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

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
        self.data = np.array(
            np.concatenate(
                [(1.0 + 0.0j, 0.0 + 0.0j) for _ in range(0, size)],
                axis=0
            )
        )

    def size(self):
        return len(self.data) // 2

    def get_qubit(self, n):
        return [
            self.data[2 * n],
            self.data[2 * n + 1]
        ]

    def set_qubit(self, n, qubit):
        self.data[2 * n] = qubit[0]
        self.data[2 * n + 1] = qubit[1]

    def get_amplitudes(self):
        result = self.get_qubit(0)

        for i in range(1, self.size()):
            result = np.kron(result, self.get_qubit(i))

        return result

    def get_qubits(self):
        return self.data

    def get_probabilities(self):
        return np.abs(self.get_amplitudes()) ** 2

    def print_qubits(self, text=''):
        for i in range(self.size()):
            print(text, self.get_qubit(i))

    def print_amplitudes(self, text=''):
        print(text, self.get_amplitudes())

    def print_probabilities(self,text=''):
        print(text, self.get_probabilities())

    def copy(self):
        qubits = QuBits()
        qubits.data = self.data.copy()
        return qubits

def rz(qubits, n, phi):
    qubits = qubits.copy()
    qubit = qubits.get_qubit(n)

    qubit = np.dot(
        np.array(
            [[cos(phi/2) - 1j * sin(phi/2), 0],
                [0, cos(phi/2) + 1j * sin(phi/2)]]
        ),
        qubit
    )

    qubits.set_qubit(n, qubit)
    return qubits

def rx(qubits, n, phi):
    qubits = qubits.copy()
    qubit = qubits.get_qubit(n)

    qubit = np.dot(
        np.array(
            [[cos(phi/2), -1j * sin(phi/2)],
                [-1j * sin(phi/2), cos(phi/2)]]
        ),
        qubit
    )

    qubits.set_qubit(n, qubit)
    return qubits

def cz(qubits, n_ctrl, n_trgt):
    qubits = qubits.copy()

    control = qubits.get_qubit(n_ctrl)
    target = qubits.get_qubit(n_trgt)

    control_target = np.kron(control, target)
    control_target = np.dot(
        np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, -1]]
        ),
        control_target
    )

    control = control_target[0: 2]
    target = control_target[2: 4]

    qubits.set_qubit(n_ctrl, control)
    qubits.set_qubit(n_trgt, target)

    return qubits
    
def cn(qubits, n_ctrl, n_trgt):
    qubits = qubits.copy()

    control = qubits.get_qubit(n_ctrl)
    target = qubits.get_qubit(n_trgt)
    control_target = np.kron(control, target)

    control_target = np.dot(
        np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        ),
        control_target
    )

    control = control_target[0: 2]
    target = control_target[2: 4]

    qubits.set_qubit(n_ctrl, control)
    qubits.set_qubit(n_trgt, target)

    return qubits
    
def x(qubits, n):
    qubits = qubits.copy()
    qubit = qubits.get_qubit(n)

    qubit = np.dot(
        np.array(
            [[0, 1],
             [1, 0]]
        ),
        qubit
    )

    qubits.set_qubit(n, qubit)
    return qubits

def r(qubits, n, phi):
    qubits = qubits.copy()
    qubit = qubits.get_qubit(n)

    qubit = np.dot(
        np.array(
            [[1, 0],
             [0, np.exp(1j * phi)]]
        ),
        qubit
    )

    qubits.set_qubit(n, qubit)
    return qubits

def h(qubits, n):
    qubits = qubits.copy()
    qubit = qubits.get_qubit(n)

    qubit = np.dot(
        0.5 * np.sqrt(2) * np.array(
            [[1, 1],
             [1, -1]]
        ),
        qubit
    )

    qubits.set_qubit(n, qubit)
    return qubits

#

class I:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        return qubits.copy()

    def dump(self):
        return "I {}".format(self.n)

class RZ:
    def __init__(self, n, phi):
        self.n = n
        self.phi = phi

    def run(self, qubits):
        return rz(qubits, self.n, self.phi)

    def dump(self):
        return "RZ ({:.2f}) {}".format(self.phi, self.n)

class RX:
    def __init__(self, n, phi):
        self.n = n
        self.phi = phi

    def run(self, qubits):
        return rx(qubits, self.n, self.phi)

    def dump(self):
        return "RX ({:.2f}) {}".format(self.phi, self.n)

class CN:
    def __init__(self, n_ctrl, n_trgt):
        self.n_ctrl = n_ctrl
        self.n_trgt = n_trgt

    def run(self, qubits):
        return cn(qubits, self.n_ctrl, self.n_trgt)

    def dump(self):
        return "CN({}, {})".format(self.n_ctrl, self.n_trgt)

class CZ:
    def __init__(self, n_ctrl, n_trgt):
        self.n_ctrl = n_ctrl
        self.n_trgt = n_trgt

    def run(self, qubits):
        return cz(qubits, self.n_ctrl, self.n_trgt)

    def dump(self):
        return "CZ({}, {})".format(self.n_ctrl, self.n_trgt)

class X:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        return x(qubits, self.n)

    def dump(self):
        return "X {}".format(self.n)

class H:
    def __init__(self, n):
        self.n = n

    def run(self, qubits):
        return h(qubits, self.n)

    def dump(self):
        return "H {}".format(self.n)

class R:
    def __init__(self, n, phi):
        self.n = n
        self.phi = phi

    def run(self, qubits):
        return r(qubits, self.n, self.phi)

    def dump(self):
        return "R {:.2f} {}".format(self.phi, self.n)

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


    for qubit_id in range(0, n_qubits):
        operations.append(X(qubit_id))

    for qubit_id in range(0, n_qubits):
        operations.append(H(qubit_id))

    for qubit1_id in range(0, n_qubits):
        for qubit2_id in range(0, n_qubits):
            if qubit1_id == qubit2_id:
                continue

            #operations.append(CZ(qubit1_id, qubit2_id))
            operations.append(CN(qubit1_id, qubit2_id))
    
    for qubit_id in range(0, n_qubits):
        for angle in [0.927295, -0.927295]:
            operations.append(R(qubit_id, angle))

    #for qubit_id in range(0, n_qubits):
    #    for angle in [0.927295, -0.927295]:
    #        operations.append(RZ(qubit_id, angle))

    #for qubit_id in range(0, n_qubits):
    #    for angle in [0.927295, -0.927295]:
    #        operations.append(RX(qubit_id, angle))

    counter = 0
    min_cost = 9999
    best_solution = None
    visited = set()

    while (counter < 200000) and (empty_heap(heap) is False):
        ((cost, rnd), node) = pop_heap(heap)
        qubits = node.qubits

        for operation in operations:
            counter += 1          

            
            new_qubits = operation.run(qubits)
            new_cost = cnll(new_qubits.get_probabilities(), expected)
            
            if new_qubits in visited:
                continue
            else:
                visited.add(new_qubits)

            #print(qubits.get_probabilities())
            #print(operation.dump())
            #print(new_qubits.get_probabilities())

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

