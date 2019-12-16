from core import semantic, system_params, input_params, tools, streamlined, result

import numpy as np
import sys
import pickle
from matplotlib import pyplot as plt

VIEW = 0  # 0: xy histogram, 1: angle arrows
BINS = 10

SAMPATH = "/local/results/inp_sample/inp_sampleL.npz"
PATH = "/local/results/learnrate2/"
PRE_FILES = ["sfa2.p"]
FILES = ["inc2_eps1_39.sfa"]
FEATURES = [0,1,2,3]

PARAMETERS = system_params.SysParamSet()

sampinp = np.load(SAMPATH)
sample_sequence, sample_categories, sample_latent = sampinp['sample_sequence'], sampinp['sample_categories'], sampinp['sample_latent']


for isfa, sfaname in enumerate(FILES):
    sfapath = PATH+sfaname
    sfa = semantic.load_SFA(sfapath)

    if PRE_FILES == []:
        samb1_y = semantic.exec_SFA(sfa, sample_sequence)
    else:
        pre_name = PRE_FILES[isfa]
        pre_path = PATH + pre_name
        pre_sfa = semantic.load_SFA(pre_path)
        seq = semantic.exec_SFA(pre_sfa, sample_sequence)
        samb1_y = semantic.exec_SFA(sfa, seq)
    samb1_w = streamlined.normalizer(samb1_y, PARAMETERS.normalization)(samb1_y)
    lat_array = np.array(sample_latent)
    lat_tups = []
    for ll in sample_latent:
        lat_tups.append((ll[2], ll[3]))
    dtyp = [('cost', float), ('sint', float)]
    slat = np.array(lat_tups, dtype=dtyp)
    sortarg = np.argsort(slat, order=['cost', 'sint'])
    samb1_sorted = samb1_w[sortarg]
    slat_sorted = lat_array[sortarg]
    nframes = len(sortarg)
    nsub = nframes // 8
    fsamp = []
    asamp = []
    if VIEW == 0:
        for fi in FEATURES:
            fsamp1, asamp1 = plt.subplots(3, 3, squeeze=False, sharex=True, sharey=True)
            fsamp1.suptitle("feature {}".format(fi+1))
            fsamp.append(fsamp1)
            asamp.append(asamp1)
    elif VIEW == 1:
        for fi in FEATURES:
            fsamp1, asamp1 = plt.subplots(1, 1, squeeze=True)
            fsamp1.suptitle("feature {}".format(fi+1))
            fsamp.append(fsamp1)
            asamp.append(asamp1)
    plot_ind_list = [[2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1]]
    hist1 = []
    max1 = [0.]*len(FEATURES)
    for ti in range(8):
        sfaout1 = samb1_sorted[ti * nsub:(ti + 1) * nsub]
        lats = slat_sorted[ti * nsub:(ti + 1) * nsub]
        hist1.append(result.histogram_2d(sfaout1, lats, norm=False, bins=BINS))
        for fi in FEATURES:
            nmax = np.max(np.absolute(hist1[ti][fi]))
            if nmax > max1[fi]:
                max1[fi] = nmax
    if VIEW == 0:
        for fi in FEATURES:
            for ti in range(8):
                pli = plot_ind_list[ti]
                axa1 = asamp[fi][pli[0]][pli[1]]
                hist1[ti][fi] /= max1[fi]
                axa1.imshow(hist1[ti][fi], interpolation='none', vmin=-1, vmax=1, extent=[-1, 1, -1, 1])
                axa1.set_title(str((45 * list(range(-3, 5))[ti])) + " °")
            hist1_accu = result.histogram_2d(samb1_sorted, slat_sorted, bins=BINS)
            asamp[fi][1][1].imshow(hist1_accu[fi], interpolation='none', vmin=-1, vmax=1, extent=[-1, 1, -1, 1])
            fsamp[fi].suptitle = (sfaname + " | " + str(fi))
    elif VIEW == 1:
        dirs = []
        vals = []
        for fi in FEATURES:
            axa = asamp[fi]
            dirs.append(np.zeros((BINS, BINS)))
            vals.append(np.zeros((BINS, BINS)))
            for x in range(BINS):
                for y in range(BINS):
                    dir0 = 0
                    val0 = 0
                    for ti in range(8):
                        his = hist1[ti][fi]
                        if np.abs(his[y, x]) > val0:
                            dir0 = ti
                            val0 = his[y, x]
                    dirs[fi][y, x] = dir0
                    vals[fi][y, x] = val0
            maxval = np.max(np.abs(vals[fi]))
            for x in range(BINS):
                for y in range(BINS):
                    lookup = np.arange(-1+1/BINS, 1, 2/BINS)
                    val0 = np.abs(vals[fi][y, x] / maxval)
                    dir0 = dirs[fi][y, x]
                    dx = (1.5/BINS)*val0 * [-1/np.sqrt(2), -1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0][int(dir0)]
                    dy = (1.5/BINS)*val0 * [-1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0, -1/np.sqrt(2), -1][int(dir0)]
                    xx = lookup[x]-dx/2
                    yy = lookup[y]-dy/2
                    axa.arrow(xx, yy, dx, dy, head_width=0.1/BINS, head_length=0.2/BINS, fc='k', ec='k')
            axa.set_xlim(-1, 1)
            axa.set_ylim(-1, 1)
            fsamp[fi].suptitle = (sfaname + " | " + str(fi))
plt.show()
