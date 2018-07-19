# coding: utf-8

# Standard library imports
import math
from math import pi
import cmath
import numpy as np
import random
import re

# PyQuil imports
import pyquil.quil as pq
from pyquil.quil import Program
from pyquil.quilbase import DefGate
from pyquil.gates import *

from referenceqvm.api import QVMConnection
from pyquil.api.qvm import QVMConnection as RQVMConnection
from pyquil.api import get_devices as Rget_devices, CompilerConnection

from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
import pyemd

import itertools
from utils import *




ROWS, COLS = 2,2
N = ROWS*COLS

REF_DISTRIBUTION = bs_dist(ROWS, COLS)

# Ease
EASE = 0.0
REF_DISTRIBUTION += EASE
REF_DISTRIBUTION /= np.sum(REF_DISTRIBUTION)

#print(REF_DISTRIBUTION)

plt.bar(range(len(REF_DISTRIBUTION)), REF_DISTRIBUTION)
plt.show()
#plt.plot(range(len(REF_DISTRIBUTION)), np.cumsum(REF_DISTRIBUTION))
#plt.show()

BUILDING_BLOCKS = {
    'I': lambda N, A: [I(i) for i in range(N)],
    'RX1': lambda N, A: [RX1(i) for i in range(N)],
    'RXX': lambda N, A: [RXX(next(A), i) for i in range(N)],
    'RZZ': lambda N, A: [RZZ(next(A), i) for i in range(N)],
    'A1': lambda N, A: [[RX1(i), RZZ(next(A), i), CZ(i, j), RX3(j)] for i in range(N) for j in range(N) if i<j]
}

def build_bb(*bb_seq):
    def fn(*A_):
        A = iter(A_)

        return pq.Program(
            [
                BUILDING_BLOCKS[bb](N, A)
                for bb in bb_seq
            ]
        )
    return fn

def count_vars_bb(*bb_seq):
    A = iter(range(9999))
    _ = [
                BUILDING_BLOCKS[bb](N, A)
                for bb in bb_seq
            ]
    return next(A)

SEQ = ['I', 'RX1', 'RZZ', 'RX1', 'RZZ', 'A1']
test1 = build_bb(*SEQ)
num_vars = count_vars_bb(*SEQ)
print('SEQ:', '-'.join(SEQ), ', vars = ', num_vars)


task = OptTask(test1, REF_DISTRIBUTION)
result = differential_evolution(task, [(-1.3, +1.3)]*num_vars, disp=True, maxiter=3, popsize=15, recombination=0.7,
                                strategy='best1bin', polish=USE_WAVEFUNCTION)

print(result)

p_best = test1(*result.x)
di = get_distribution(p_best)
print(di)
print('KLdiv', kl_div(di, REF_DISTRIBUTION))
print('CNLL', cnll(di, REF_DISTRIBUTION))
print('EMD', emd(di, REF_DISTRIBUTION))

qvm = QVMConnection()
wf = qvm.wavefunction(p_best)[0]
wf.plot()


# In[65]:


probs = np.square(np.abs(wf.amplitudes))
plt.bar(range(len(REF_DISTRIBUTION)), REF_DISTRIBUTION, width=0.8)
plt.bar(range(len(REF_DISTRIBUTION)), probs, width=0.6)
plt.bar(range(len(REF_DISTRIBUTION)), artificial_sample(probs, samples=100), width=0.2)
plt.show()

#print(task.history)
H = np.array(task.history)
Hmin = np.minimum.accumulate(H)

# ax = plt.gca()
# ax.scatter(range(len(task.history)), task.history, 2, 'red')
# ax.plot(range(len(task.history)), Hmin, linewidth=1, color='black')
# ax.set_yscale('log')
# plt.show()

dev = Rget_devices(as_dict=True)['8Q-Agave']
comp = CompilerConnection(device=dev)
cpp = comp.compile(p_best)

qvm = QVMConnection()
rwf = qvm.wavefunction(p_best)[0]
rwf.plot()

