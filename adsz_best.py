import os
from collections import defaultdict

DATA = defaultdict(lambda: defaultdict(list))

for root, dirs, files in os.walk('adsz_runs'):
    if 'cnll.score' in files:
        with open(os.path.join(root, 'cnll.score')) as f:
            score = float(f.read())

        adsz_runs, rc, seq, tim = root.split('/')

        DATA[rc][seq].append(score)

for rc, data_for_rc in DATA.items():

    print('#' + rc)

    ITEMS = list(data_for_rc.items())
    ITEMS = sorted(ITEMS, key=lambda kv: min(kv[1]))
    for seq, items in ITEMS:
        print(seq)
        print(items)