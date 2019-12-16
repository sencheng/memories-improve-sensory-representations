from core import tools, result

import numpy as np

from matplotlib import pyplot

PATH = "../results/sfaonelayer/sfaonelayer/"
PREFIX = "res"
N = 18
ROWS = 2
COLUMNS = N/ROWS
POSTFIX = ".p"
DKEYS = ['sfa1', 'sfa2S', 'sfa2E', 'forming_Y', 'retrieved_Y', 'testingY', 'testingZ_S', 'testingZ_E']
DCOLORS = ['r','r','r','y','y','k','k','k']

res_list = []
f, axarr = pyplot.subplots(ROWS, COLUMNS, sharex=True, sharey=True)

for n in range(N):
    res_list.append(result.load_from_file(PATH+PREFIX+str(n)+POSTFIX))
    ax = axarr[n/COLUMNS][n%COLUMNS]
    ax.set_title(str(n))
    for i,(k,c) in enumerate(zip(DKEYS,DCOLORS)):
        if Exception == type(res_list[n]):
            ax.text(len(DKEYS)/2, 0, res_list[n], ha='center', va='bottom', color='black')
        else:
            d = np.mean(res_list[n].d_values[k][:8])  # mean of first 8 d-values
            ax.bar([i + 0.1], [d], width=0.8, color=c)
            ax.text(i + 0.5, d, '{:.4f}'.format(d), ha='center', va='bottom', color="black", rotation=90)
    tix = list(np.arange(len(DKEYS))+0.5)
    ax.set_xticks(tix)
    ax.set_xticklabels(DKEYS, rotation = 70)
    ax.set_yscale("log",nonposy='clip')