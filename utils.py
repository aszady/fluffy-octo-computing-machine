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


# eg. 6 -> '011'  (3qbits)
def le_to_state(i, qbits_num):
    fmt = '{{:0{}b}}'.format(qbits_num)
    return fmt.format(i)[::-1]


assert (le_to_state(6, 3) == '011')


# eg. '110' -> 6
def state_to_be(state):
    return int(state, 2)


assert (state_to_be('110') == 6)


# eg. 6 <-> 3  (3qbits)
def swap_endianness(i, qbits_num):
    return state_to_be(le_to_state(i, qbits_num))


assert (swap_endianness(6, 3) == 3)
assert (swap_endianness(3, 3) == 6)


# eg. 6 -> (0, 1, 1)  (3qbits)
def le_to_measurement(i, qbits_num):
    return tuple(map(int, le_to_state(i, qbits_num)))


assert (le_to_measurement(6, 3) == (0, 1, 1))


def format_program(program):
    return re.sub(
        r'(DEFGATE [a-zA-Z0-9_]+:)\n((    .*\n)+)\n',
        r'\1<div class="matrix">\2</div>\n',
        str(program),
        flags=re.M | re.I)


def label_be(measurement):
    state_fmt = ''
    for qbit_id, bit in enumerate(measurement):
        state_fmt += '{}<sub>{}</sub>'.format(bit, qbit_id)
    return state_fmt


def format_html(amplitudes, measurements, program, label, show_quil):
    qbits_num = int(math.log2(len(amplitudes)))

    lines = []

    le_amplitudes = enumerate(amplitudes)
    be_amplitudes = {swap_endianness(i_le, qbits_num): amplitude for i_le, amplitude in le_amplitudes}

    for i_be, amplitude in sorted(be_amplitudes.items()):
        i_le = swap_endianness(i_be, qbits_num)
        state = le_to_state(i_le, qbits_num)

        mod = abs(amplitude)
        hue = cmath.phase(amplitude) / 2 / math.pi * 360

        measurement = le_to_measurement(i_le, qbits_num)

        lines.append('''<tr>
            <td class="be">{i_be}</td>
            <td class="s">|{state}⟩</td>
            <td class="a"><div class="wr"><div style="background: linear-gradient(hsl({hue}, 60%, 50%), hsl({hue}, 90%, 30%)); width: {width}%;"></div></div></td>
            <td class="ar {zero_class}">{areal:+.04f}</td>
            <td class="ai {zero_class}">{aimag:+.04f}ⅈ</td>
            </tr>'''.format(
            i_be=i_be,
            state=label(measurement),
            zero_class='zero' if mod == 0 else '',
            areal=amplitude.real,
            aimag=amplitude.imag,
            hue=hue,
            width=mod * 100
        )
        )

    return '''<style>
        .amps{border:1px solid #000; border-collapse:collapse; font-family: monospace}
        .amps tr { border: 1px solid #111 }
        .amps tr th { text-align: center }
        .amps td { padding: 0.4em; vertical-align: middle; }
        .amps td.a { width: 10em; }
        .amps td .wr { border: 1px solid #555; height: 1em; width: 100%; box-sizing: border-box; background: #eee; }
        .amps td .wr div { height:100%; }
        .amps td.m { width: 5em; }
        .amps td.m .wr div { background: black; }
        .amps td.ar, .amps td.ai { font-family: Arial; font-size: 0.9em; }
        .amps td.ar.zero, .amps td.ai.zero { opacity: 0.5; }
        .amps td.s, .amps td.ai { border-right: 1px solid #333; }
        .amps td sub { opacity: 0.5; color: #024; font-size:0.5em; }
        .result { display: table-row; border-collapse: collapse;}
        .result pre, result div.right { display: table-cell }
        .result pre { background: linear-gradient(#fdfdb1, #dede83) !important;
            padding: 1em !important; padding-right: 5em !important; border: 1px solid black !important;
            border-right: 0px !important; min-width: 15em; max-width: 30em; }
        .result pre .matrix { font-size: .75em; line-height: 1em; }
    </style>
    <div class="result">''' + \
           ('<pre>{program}</pre>'.format(program=format_program(program)) if show_quil else '') + \
           '''<div class="right">
           <table class="amps">
           <tr>
           <th colspan="2">State</th>
           <th colspan="3">Amp (1 run)</th>
           </tr>'''.format(program=format_program(program)) + \
           '\n'.join(lines) + \
           '''</table></div></div>'''


def run_program(*instructions, **kwargs):
    trials = kwargs.get('trials', 1)
    quiet = kwargs.get('quiet', False)
    label = kwargs.get('label', label_be)
    quil = kwargs.get('quil', True)

    program = pq.Program(*instructions)

    qvm = QVMConnection()
    measurements = qvm.run_and_measure(program, list(program.get_qubits()), trials=trials)
    amplitudes = qvm.wavefunction(program)[0].amplitudes

    if not quiet:
        display(HTML(format_html(amplitudes, measurements, program, label=label, show_quil=quil)))

    return qvm.classical_memory


def bars_and_stripes(rows, cols):
    data = []

    for h in itertools.product([0, 1], repeat=cols):
        pic = np.repeat([h], rows, 0)
        data.append(pic.ravel().tolist())

    for h in itertools.product([0, 1], repeat=rows):
        pic = np.repeat([h], cols, 1)
        data.append(pic.ravel().tolist())

    data = np.unique(np.asarray(data), axis=0)

    return data


def bs_dist(rows, cols):
    bas = bars_and_stripes(rows, cols)

    n_points, n_qubits = bas.shape

    fig, ax = plt.subplots(1, bas.shape[0], figsize=(9, 1))
    for i in range(bas.shape[0]):
        ax[i].matshow(bas[i].reshape(rows, cols), vmin=-1, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    '''sample distribution'''
    hist_sample = [0 for _ in range(2 ** n_qubits)]
    for s in bas:
        b = ''.join(str(int(e)) for e in s)
        idx = int(b, 2)
        hist_sample[idx] += 1. / float(n_points)

    plt.show()

    return np.array(hist_sample)


QVM_CXN = QVMConnection()
USE_WAVEFUNCTION = True


def get_distribution(p):
    wf = QVM_CXN.wavefunction(p)
    return np.square(np.abs(wf[0].amplitudes))  # local
    # return np.square(np.abs(wf.amplitudes)) # Rigetti


def kl_div(p, q, eps=1e-8):
    n = len(p)
    assert n == len(q)

    p = np.array(p)
    q = np.array(q)

    return np.sum(p * np.log((eps + p) / (eps + q)))


def cnll(p, q, eps=1e-8):
    return -np.sum(np.log(np.maximum(p, eps)) * q) / np.sum(q)


def emd(p, q):
    x = np.arange(len(p))
    dst = np.square(np.abs(np.array(x[:, np.newaxis]) - np.array(x[np.newaxis, :])).astype(float))
    return pyemd.emd(p, q, dst)


def artificial_sample(probs, samples):
    cumprobs = np.cumsum(probs)
    randoms = np.sort(np.random.random(samples))
    results = np.zeros_like(probs)

    sample_id = 0
    for state_id in range(len(cumprobs)):
        while sample_id < samples and randoms[sample_id] < cumprobs[state_id]:
            results[state_id] += 1
            sample_id += 1
    results /= samples

    return results


class OptTask:
    def __init__(self, p, ref_distribution, metric=kl_div):
        self.p = p
        self.ref_distribution = ref_distribution
        self.history = []
        self.cache = {}
        self.metric = metric

    def __call__(self, X):
        xt = tuple(X)
        if xt in self.cache:
            return self.cache[xt]

        di = get_distribution(self.p(*X))
        if not USE_WAVEFUNCTION:
            di = artificial_sample(di, 100)
        div = self.metric(di, self.ref_distribution)
        self.history.append(div)

        self.cache[xt] = div
        # print('{} -> {}'.format(X, div))
        return div

RX1 = lambda x: RX(pi/2, x)
RX2 = lambda x: RX(pi, x)
RX3 = lambda x: RX(-pi/2, x)
RZZ = lambda a, x: RZ(a*pi, x)