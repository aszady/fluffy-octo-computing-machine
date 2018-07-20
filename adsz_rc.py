# coding: utf-8

# Standard library imports
import math
from functools import partial
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

def BRX1(N): return lambda A: [RX1(i) for i in range(N)]
def BRX2(i): return lambda A: RX2(i)
def BRX3(i): return lambda A: RX3(i)
def BRZZ(N): return lambda A: [RZZ(next(A), i) for i in range(N)]
def BCZ(i, j): return lambda A: CZ(i, j)

def BA5(i, j): return lambda A: [RZZ(next(A), i), RZZ(next(A), j), RX1(i), RX1(j), CZ(i, j)]
def BA6(i, j): return lambda A: [RZZ(next(A), j), RX1(i), CZ(i, j)]

BUILDING_BLOCKS = dict(\
    [('RX1', BRX1(N))] +\
    #[('RX2.{}'.format(i), BRX2(i)) for i in range(N)] +\
    #[('RX3.{}'.format(i), BRX3(i)) for i in range(N)] +\
    [('RZZ', BRZZ(N))] +\
    [('CZ.{}.{}'.format(i, j), BCZ(i, j))
      for i in range(N) for j in range(N) if (i,j) in TOPO] +\
    [('A5.{}.{}'.format(i, j), BA5(i, j))
      for i in range(N) for j in range(N) if (i, j) in TOPO]+\
    [('A6.{}.{}'.format(i, j), BA6(i, j))
      for i in range(N) for j in range(N) if (i, j) in TOPO]
    )


def build_bb(*bb_seq):
    def fn(*A_):
        A = iter(A_)

        return pq.Program(
            [I(i) for i in range(N)],
            [
                BUILDING_BLOCKS[bb](A)
                for bb in bb_seq
            ]
        )
    return fn

def count_vars_bb(*bb_seq):
    A = iter(range(9999))
    _ = [
                BUILDING_BLOCKS[bb](A)
                for bb in bb_seq
            ]
    return next(A)

PROJECT_NAME = str(args.t.lower()) + '-' + str(ROWS) + '-' + str(COLS) + '/RC/' + str(int(time.time()))
dir = os.path.join(os.path.dirname(__file__), 'adsz_runs', PROJECT_NAME)
os.makedirs(dir, exist_ok=True)

DATA = [{
    'seq': ['RX1'] +\
           ['RZZ'] +\
            ['RX1'] +\
            ['A5.{}.{}'.format(i,j) for i in range(N) for j in range(N) if (i,j) in TOPO if i<j]
    ,
    'score': 10000
}]

best_score = 1000
best_count = 0

while True:
    DATA = sorted(DATA, key=lambda x: (x['score'] + len(x['seq'])*0.01))[:20]

    prev = random.choices(DATA, weights=np.linspace(1, 0, len(DATA)))[0]

    print('## Ranking ##')
    for d in reversed(DATA):
        print(d['score'], len(d['seq']), '-'.join(d['seq']))
    print('## ^^ Top ##')

    print('Prev:', prev)

    ne = {
        'seq': prev['seq'][:]
    }

    action = 'add'
    if len(ne['seq']) >= 1 and random.random() < 0.5:
        action = 'del'

    if action == 'add':
        for z in range(5):
            p = random.randint(0, len(ne['seq']))
            ne['seq'].insert(p, random.choice(list(BUILDING_BLOCKS.keys())))
    elif action == 'del':
        p = random.randint(0, len(prev['seq']) - 1)
        ne['seq'] = ne['seq'][:p] + ne['seq'][p+1:]

    test1 = build_bb(*ne['seq'])
    num_vars = count_vars_bb(*ne['seq'])


    print('New seq: ', ' '.join(ne['seq']), ', num_vars = ', num_vars)

    task = OptTask(test1, REF_DISTRIBUTION)

    result = None
    if num_vars == 0:
        print('No vars, calling directly')
        print(test1())
        ne['score'] = task([])
    else:
        result = differential_evolution(task, [(-1.3, +1.3)] * num_vars, init='random',
            disp=True, maxiter=3, popsize=5, recombination=0.7,
            strategy='best1bin', polish=USE_WAVEFUNCTION)

        ne['score'] = result.fun

    DATA.append(ne)
    print('Done')

    if ne['score'] < best_score:
        best_score = ne['score']
        best_count += 1
        print('New best score!')

        p_best = test1(*result.x if num_vars > 0 else [])

        qvm = QVMConnection()
        wf = qvm.wavefunction(p_best)[0]

        plt.cla()
        probs = np.square(np.abs(wf.amplitudes))
        plt.bar(range(len(REF_DISTRIBUTION)), REF_DISTRIBUTION, width=0.8)
        plt.bar(range(len(REF_DISTRIBUTION)), probs, width=0.6)
        #plt.bar(range(len(REF_DISTRIBUTION)), artificial_sample(probs, samples=100), width=0.2)
        plt.title('-'.join(ne['seq']))
        plt.savefig(os.path.join(dir, '{}-chart1.png'.format(best_count)), dpi=300)

        H = np.array(task.history)
        Hmin = np.minimum.accumulate(H)
        plt.cla()
        fig = plt.gcf()
        ax = fig.gca()
        ax.scatter(range(len(task.history)), task.history, 2, 'red')
        ax.plot(range(len(task.history)), Hmin, linewidth=1, color='black')
        ax.set_yscale('log')
        fig.savefig(os.path.join(dir, '{}-chart2.png'.format(best_count)))

        with open(os.path.join(dir, '{}-recipe.txt'.format(best_count)), 'wt') as f:
            f.write(str('-'.join(ne['seq'])))
            f.write('\n')
            f.write(str(ne['score']))

        with open(os.path.join(dir, '{}-log.txt'.format(best_count)), 'wt') as f:
            f.write(str(p_best))
            f.write('Compiled:\n')
            try:
                f.write(str(postoptimize(p_best)))
            except Exception:
                print('NOT writing compiled version :(')
                pass

