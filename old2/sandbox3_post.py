import pickle
import matplotlib.pyplot as plt
import numpy as np

res = pickle.load(open("/local/results/sb3.p", 'rb'))
dia = res.dmat_dia
hst = plt.hist(dia, 100)

maxx = np.max(hst[1])
maxy = np.max(hst[0])

perc = res.params.st2['memory']['smoothing_percentile']
x = np.percentile(dia, perc)

plt.plot((x,x), (0, maxy), 'r')

res.plot_delta()
