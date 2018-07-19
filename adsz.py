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
from pyquil.gates import *

from referenceqvm.api import QVMConnection
from pyquil.api.qvm import QVMConnection as RQVMConnection
from pyquil.api import get_devices as Rget_devices, CompilerConnection

from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
import time
import os
import argparse
import itertools

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('rows', type=int, default=2)
parser.add_argument('cols', type=int, default=2)
parser.add_argument('-k', type=int, default=3)
parser.add_argument('-seq', type=str)
parser.add_argument('-t', type=str, default='all')
args = parser.parse_args()

ROWS, COLS = args.rows, args.cols
K = args.k
N = ROWS*COLS
TOPO = topology(N, args.t)

REF_DISTRIBUTION = bs_dist(ROWS, COLS)

# Ease
EASE = 0.0
REF_DISTRIBUTION += EASE
REF_DISTRIBUTION /= np.sum(REF_DISTRIBUTION)

INIT_ARR = np.zeros(2**N)
INIT_ARR[0] = 1.

#print(REF_DISTRIBUTION)

# plt.bar(range(len(REF_DISTRIBUTION)), REF_DISTRIBUTION)
# plt.show()

BUILDING_BLOCKS = {
    #'I': lambda N, A: [I(i) for i in range(N)],
    #'RX1': lambda N, A: [RX1(i) for i in range(N)],
    'RX1y': lambda N, A: [RX1(i) for i in range(1,N)],
    #'RXX': lambda N, A: [RXX(next(A), i) for i in range(N)],
    #'RZZ': lambda N, A: [RZZ(next(A), i) for i in range(N)],
    'RZZy': lambda N, A: [RZZ(next(A), i) for i in range(1, N)],
    #'A1': lambda N, A: [[RX1(i), RZZ(next(A), i), CZ(i, j), RX3(j)] for i in range(N) for j in range(N) if i<j],
    #'A2': lambda N, A: [[RX1(i), RZZ(next(A), i), CZ(i, j), RX3(j)] for i in range(N) for j in range(N) if i!=j],
    #'A3': lambda N, A: [[RX1(i), RZZ(next(A), j), CZ(i, j), RX3(j)] for i in range(N) for j in range(N) if i!=j],
    #'A4': lambda N, A: [[RZZ(next(A), i), RZZ(next(A), j), RX1(i), CZ(i, j)] for i in range(N) for j in range(1,N) if i!=j],
    'A4y': lambda N, A: [[RZZ(next(A), i), RZZ(next(A), j), RX1(i), CZ(i, j)]
                         for i in range(1,N) for j in range(1,N) if (i,j) in TOPO],
    'A5y': lambda N, A: [[RZZ(next(A), i), RX1(i), RZZ(next(A), j), RX1(j), CZ(i, j)] for i in range(1,N) for j in range(1,N) if (i,j) in TOPO],
    #'A4S': lambda N, A: [[RZZ(next(A), i), RZZ(next(A), (i+1)%N), RX1(i), CZ(i, (i+1)%N)] for i in range(N)],
    'SY': lambda N, A: [RX1(0)] + [CNOT(0, i) for i in range(1,N)]
}


def is_entangling(name):
    return 'A' in name

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


begin_with_rx1 = random.choice([False, True])

if args.seq:
    SEQ = args.seq.split('-')
else:
    SEQ = []
    if begin_with_rx1:
        SEQ.append('RX1y')

    while True:
        RSEQ = random.choices(population=list(
            set(BUILDING_BLOCKS.keys()) - {'SY'}),
            k=args.k - len(SEQ) - 1)
        if any(map(is_entangling, RSEQ)):
            break

    SEQ.extend(RSEQ)
    SEQ.append('SY')


PROJECT_NAME = str(args.t.lower()) + '-' + str(ROWS) + '-' + str(COLS) + '/' + '-'.join(SEQ) + '/' + str(int(time.time()))
dir = os.path.join(os.path.dirname(__file__), 'adsz_runs', PROJECT_NAME)
os.makedirs(dir, exist_ok=True)
logfile = open(os.path.join(dir, 'log.txt'), 'at')

def LOG(*args):
    s = ' '.join(map(str, args))
    print(s)
    logfile.write(s + '\n')
    logfile.flush()

test1 = build_bb(*SEQ)
num_vars = count_vars_bb(*SEQ)
num_gates = len(test1(*[0.12]*num_vars).instructions)

LOG(PROJECT_NAME)
LOG('SEQ:', '-'.join(SEQ), ', vars = ', num_vars, ', num_gates = ', num_gates)


task = OptTask(test1, REF_DISTRIBUTION)
result = differential_evolution(task, [(-1.3, +1.3)]*num_vars, init='random',
    disp=True, maxiter=2, popsize=10, recombination=0.7,
    strategy='best1bin', polish=USE_WAVEFUNCTION)

LOG(result)

p_best = test1(*result.x)
di = get_distribution(p_best)
LOG(di)
cnll_score = cnll(di, REF_DISTRIBUTION)
LOG('KLdiv', kl_div(di, REF_DISTRIBUTION))
open(os.path.join(dir, 'cnll.score'), 'wt').write(str(cnll_score))

LOG('CNLL', cnll_score)
LOG('EMD', emd(di, REF_DISTRIBUTION))

LOG(p_best)

qvm = QVMConnection()
wf = qvm.wavefunction(p_best)[0]
#wf.plot()


# In[65]:

plt.cla()
probs = np.square(np.abs(wf.amplitudes))
plt.bar(range(len(REF_DISTRIBUTION)), REF_DISTRIBUTION, width=0.8)
plt.bar(range(len(REF_DISTRIBUTION)), probs, width=0.6)
plt.bar(range(len(REF_DISTRIBUTION)), artificial_sample(probs, samples=100), width=0.2)
plt.title('-'.join(SEQ))
plt.savefig(os.path.join(dir, 'chart1.png'))

#print(task.history)
H = np.array(task.history)
Hmin = np.minimum.accumulate(H)

plt.cla()
fig = plt.gcf()
ax = fig.gca()
ax.scatter(range(len(task.history)), task.history, 2, 'red')
ax.plot(range(len(task.history)), Hmin, linewidth=1, color='black')
ax.set_yscale('log')
fig.savefig(os.path.join(dir, 'chart2.png'))

# dev = Rget_devices(as_dict=True)['8Q-Agave']
# comp = CompilerConnection(device=dev)
# cpp = comp.compile(p_best)
#
# qvm = QVMConnection()
# rwf = qvm.wavefunction(p_best)[0]
# rwf.plot()

