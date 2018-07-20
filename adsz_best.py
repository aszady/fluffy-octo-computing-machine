import os
from collections import defaultdict

DATA = defaultdict(lambda: defaultdict(list))

for root, dirs, files in os.walk('adsz_runs'):
    if 'cnll.score' in files:
        with open(os.path.join(root, 'cnll.score')) as f:
            score = float(f.read())

        adsz_runs, rc, seq, tim = root.split('/')

        num_lines = 0
        with open(os.path.join(root, 'log.txt')) as f:
            lines = f.readlines()
            if 'Compiled:\n' in lines:
                p = lines.index('Compiled:\n')
                lines = lines[p+1:]
                lines = list(filter(lambda x: x.find('PRAGMA') == -1 and len(x) > 1, lines))
                num_lines = len(lines)

        DATA[rc][seq].append((score, num_lines))

        #lines.index('Compiled:\n')

def score_sample(s):
    time, depth = s
    if depth == 0:
        depth = 9999
    return time + depth*0.001

def score_item(a):
    samples = a[1]
    return min(map(score_sample, samples))

for rc, data_for_rc in DATA.items():

    print('#' + rc)

    ITEMS = list(data_for_rc.items())
    ITEMS = sorted(ITEMS, key=score_item)
    for seq, items in ITEMS[:5]:
        print('{:20s}'.format(seq), items)

    print()