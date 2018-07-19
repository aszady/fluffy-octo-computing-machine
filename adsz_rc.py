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
args = parser.parse_args()

ROWS, COLS = args.rows, args.cols
K = args.k

N = ROWS*COLS

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

def BRX1(i): return lambda A: RX1(i)
def BRX2(i): return lambda A: RX2(i)
def BRX3(i): return lambda A: RX3(i)
def BRZZ(i): return lambda A: RZZ(next(A), i)
def BCZ(i, j): return lambda A: CZ(i, j)

BUILDING_BLOCKS = dict(\
    [('RX1.{}'.format(i), BRX1(i)) for i in range(N)] +\
    [('RX2.{}'.format(i), BRX2(i)) for i in range(N)] +\
    [('RX3.{}'.format(i), BRX3(i)) for i in range(N)] +\
    [('RZZ.{}'.format(i), BRZZ(i)) for i in range(N)] +\
    [('CZ.{}.{}'.format(i, j), BCZ(i, j))
      for i in range(N) for j in range(N) if i!=j])


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

DATA = [{
    'seq': ['RX1.{}'.format(i) for i in range(N)] +\
           ['RZZ.{}'.format(i) for i in range(N)] +\
            ['RX1.{}'.format(i) for i in range(N)]
    ,
    'score': 10000
}]

while True:
    DATA = sorted(DATA, key=lambda x: (x['score'], x['seq']))[:20]


    prev = random.choices(DATA, weights=np.linspace(1, 0, len(DATA)))[0]

    print('## Ranking ##')
    for d in reversed(DATA):
        print(d['score'], len(d['seq']), '-'.join(d['seq']))
    print('## ^^ Top ##')

    print('Prev:', prev)

    ne = {
        'seq': prev['seq'][:]
    }

    p = random.randint(0, len(prev['seq']))
    ne['seq'].insert(p, random.choice(list(BUILDING_BLOCKS.keys())))

    test1 = build_bb(*ne['seq'])
    num_vars = count_vars_bb(*ne['seq'])


    print('New seq: ', ' '.join(ne['seq']), ', num_vars = ', num_vars)

    task = OptTask(test1, REF_DISTRIBUTION)

    if num_vars == 0:
        print('No vars, calling directly')
        print(test1())
        ne['score'] = task([])
    else:
        result = differential_evolution(task, [(-1.3, +1.3)] * num_vars, init='random',
            disp=False, maxiter=3, popsize=5, recombination=0.7,
            strategy='best1bin', polish=USE_WAVEFUNCTION)

        ne['score'] = result.fun

    DATA.append(ne)
    print('Done')

