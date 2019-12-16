import numpy as np
import pickle
from matplotlib import pyplot as plt

PATH = "/local/results/reorder_norepa/"

files = []
dists = []
f, ax = plt.subplots(3, 3, sharex=True, sharey=True)
for i in range(9):
    with open(PATH+"res{}.p".format(i), 'rb') as f:
        res = pickle.load(f)
        files.append(res)
    # numin = len(res.dmat_dia)
    numin = 60000
    dist = np.zeros(numin)
    for numb in res.retrieved_indices:
        dist[numb] += 1
    dists.append(dist)

    row = i // 3
    col = i % 3
    axx = ax[col, row]
    y = np.sort(dist)
    x = range(numin)
    axx.plot(x,y)
    axx.set_title(i)

plt.show()
