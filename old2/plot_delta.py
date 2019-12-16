from core import semantic

import numpy as np
from matplotlib import pyplot
import os

DIR = "/home/goerlrwh/Dropbox/arbeit/Semantic-Episodic/17-02-07 results/sandbox4/"
ROWS = 3
COLUMNS = 1

D_FILE = True

if not D_FILE:
    ss = semantic.SemanticSystem()

    f, axarr = pyplot.subplots(ROWS, COLUMNS, sharex=True, sharey=True)
    for n, filename in enumerate(iter(sorted(list(os.listdir(DIR))))):
        ax = axarr[n / COLUMNS][n % COLUMNS]
        ax.set_title(filename)
        sfa = ss.load_SFA_module(DIR+filename)
        d = semantic.get_d_values(sfa,get_all=True)
        d_mean = [np.mean(x) for x in d[:16]]
        print(d_mean)
        for i, dd in enumerate(d_mean):
            ax.bar([i + 0.1], [dd], width=0.8, color='k')
            ax.text(i + 0.5, dd, '{:.3f}'.format(dd), ha='center', va='bottom', color="black")
            #ax.set_yscale("log", nonposy='clip')
        ax.text(len(d_mean) / 2, max(d_mean) / 2, 'SFA2 eff.: ' + '{:.3f}'.format(d_mean[-3] / d_mean[-1]), ha='center', va='center', color="black")
else:
    d_array = np.load(DIR+"d.npy")
    f, axarr = pyplot.subplots(ROWS, COLUMNS, sharex=True, sharey=True)
    if d_array.shape[1] == 1:   # only one input type
        for n, d in enumerate(d_array):   #d - sfa type
            if COLUMNS == 1 or ROWS == 1:
                ax = axarr[n]
            else:
                ax = axarr[n / COLUMNS][n % COLUMNS]
            ax.set_title(str(n))
            d_mean = [np.mean(x) for x in [arr[:16] for arr in d[0,0]]]     # [input-type, sfapart(bool)]. d[0,0] is the list of sfa layers for sfa part 1
            print(d_mean)
            sfa1_out = 0
            sfa1_layers = len(d_mean)
            dmax = max(d_mean)
            for i, dd in enumerate(d_mean):
                if i == sfa1_layers-1:
                    sfa1_out=dd
                ax.bar([i + 0.1], [dd], width=0.8, color='b')
                ax.text(i + 0.5, dd, '{:.3f}'.format(dd), ha='center', va='bottom', color="black")
                # ax.set_yscale("log", nonposy='clip')

            d_mean = [np.mean(x) for x in [arr[:16] for arr in d[0,1]]]  # now sfa part 2
            print(d_mean)
            sfa2_out = 0
            for i, dd in enumerate(d_mean):
                if i == len(d_mean)-1:
                    sfa2_out = dd
                ax.bar([i+sfa1_layers + 0.1], [dd], width=0.8, color='k')
                ax.text(i+sfa1_layers + 0.5, dd, '{:.3f}'.format(dd), ha='center', va='bottom', color="black")
            ax.text(sfa1_layers+len(d_mean)/2, dmax/2, 'SFA2 eff.: ' + '{:.3f}'.format(sfa1_out/sfa2_out), ha='center', va='center', color="black")

    else:
        for col, dc in enumerate(d_array):      #dc - sfa type
            for row, d in enumerate(dc):        #d - input type
                ax = axarr[row][col]
                ax.set_title("row " + str(row) + "col" + str(col))
                d_mean = [np.mean(x) for x in [arr[:16] for arr in d[0]]]
                print(d_mean)
                sfa1_out = 0
                sfa1_layers = len(d_mean)
                dmax = max(d_mean)
                for i, dd in enumerate(d_mean):
                    if i == sfa1_layers - 1:
                        sfa1_out = dd
                    ax.bar([i + 0.1], [dd], width=0.8, color='k')
                    ax.text(i + 0.5, dd, '{:.3f}'.format(dd), ha='center', va='bottom', color="black")
                    # ax.set_yscale("log", nonposy='clip')

                    d_mean = [np.mean(x) for x in [arr[:16] for arr in d[1]]]    # sfa2
                print(d_mean)
                sfa2_out = 0
                for i, dd in enumerate(d_mean):
                    if i == len(d_mean) - 1:
                        sfa2_out = dd
                    ax.bar([i+sfa1_layers + 0.1], [dd], width=0.8, color='k')
                    ax.text(i+sfa1_layers + 0.5, dd, '{:.3f}'.format(dd), ha='center', va='bottom', color="black")
                ax.text(sfa1_layers+len(d_mean)/2, dmax/2, 'SFA2 eff.: ' + '{:.3f}'.format(sfa1_out/sfa2_out), ha='center', va='center', color="black")

pyplot.show()