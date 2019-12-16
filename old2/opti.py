import pickle
import numpy as np
from matplotlib import pyplot

# Script for loading result file containing opti_values and processing and plotting them in a convenient way

STOP = 10

path = "/home/goerlrwh/Dropbox/arbeit/Semantic-Episodic/17-01-30 results/"

res = pickle.load(open(path+"test1d_c.p"))

o = [[]]
s = []

for el in res.opti_values:
    if not el[0] == 10e10:
        o[-1].append(np.array(el))
    else:
        s.append(np.array(el[1]))
        o.append([])

del o[-1]

sequences = np.array(o)
shapes = np.array(s)
sh = shapes[0, 0]

mean_radius = np.mean(sequences[:,:,0])
mean_fraction = np.mean(sequences[:,:,1]/sh)
print("Overall mean radius: {}, Overall mean fraction: {}".format(mean_radius, mean_fraction))

for i, seq in enumerate(sequences):
    if i > STOP:
        break
    x = range(len(seq))
    y1 = []
    y2 = []
    for el in seq:
        radius = el[0]
        index_range = el[1]
        y1.append(radius)
        y2.append(index_range/sh)
    pyplot.subplot(1,2,1)
    pyplot.plot(x,y1)
    pyplot.text(0, max(y1), "mean {}".format(np.mean(y1)))
    pyplot.subplot(1,2,2)
    pyplot.plot(x,y2)
    pyplot.text(0, max(y2), "mean {}".format(np.mean(y2)))
    pyplot.show()
