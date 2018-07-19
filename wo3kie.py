from utils import cnll

from heapq import *
from math import log, pi, sin, cos
import random

import copy
import numpy as np
import os

from pyquil.api import QVMConnection
from pyquil.quil import Program
from pyquil.gates import *

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
        return qubits

    def get_gate(self):
        return gate

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

def rz(qubits: QuBits, n, phi):
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

with open(os.path.expanduser('~/.pyquil_config'), 'w') as f:
    f.write(PYQUIL_CONFIG)

qvm = QVMConnection()
pro = Program()

#

def find_lcz(actual, expected):
    n_qubits = int(np.log2(len(actual)))
    cost = cnll(actual, expected)

    heap = make_heap()
    push_heap(heap, Node(parent=None, qubits=QuBits(4), gate=None, cost=cost), (cost, random.random()))

    operations = []

    def build_lambda_x(i):
        def fn(qs):
            return x(qubits, i)

        return fn

    for qubit_id in range(0, n_qubits):
        operations.append(build_lambda_x(qubit_id))

    def build_lambda_h(i):
        def fn(qs):
            return h(qubits, i)

        return fn

    for qubit_id in range(0, n_qubits):
        operations.append(build_lambda_h(qubit_id))

    def build_lambda_cz(i1, i2):
        def fn(qs):
            return cz(qubits, i1, i2)

        return fn

    for qubit1_id in range(0, n_qubits):
        for qubit2_id in range(0, n_qubits):
            if qubit1_id == qubit2_id:
                continue
            operations.append(build_lambda_cz(qubit1_id, qubit2_id))

    N=0

    def build_lambda_rz(i, a):
        def fn(qs):
            return rz(qubits, i, a)

        return fn

    #for qubit_id in range(0, n_qubits):
    #    for angle in [2*pi*i/N for i in range(0, N)]:
    #        operations.append(build_lambda_rz(qubit_id, angle))

    def build_lambda_rx(i, a):
        def fn(qs):
            return rx(qubits, i, a)

        return fn

    for qubit_id in range(0, n_qubits):
        for angle in [pi*i/N for i in range(0, N)]:
            operations.append(build_lambda_rx(qubit_id, angle))

    counter = 0
    min_cost = 9999
    best_solution = None
    visited = set()

    while (counter < 1000000) and (empty_heap(heap) is False):
        (cost, node) = pop_heap(heap)

        qubits = node.qubits

        for operation in operations:
            counter += 1          

            if counter % 1000 == 0:
                print(counter, end='\r')  
            
            new_qubits = operation(qubits)
            cost = cnll(qubits.get_probabilities(), expected)
            new_cost = cnll(new_qubits.get_probabilities(), expected)
            
            #print(qubits.get_probabilities())
            #print(new_qubits.get_probabilities())
            #print(expected)
            #print()

            if new_qubits in visited:
                continue

            if new_cost < min_cost:
                min_cost = new_cost
                best_solution = new_qubits
                print(cost, " -> ", new_cost)

            visited.add(new_qubits)
            push_heap(heap, Node(parent=None, qubits=new_qubits, gate=None, cost=cost), (cost, random.random()))


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
    print(find_lcz(actual, expected))

main()

