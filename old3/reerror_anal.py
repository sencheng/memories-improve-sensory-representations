"""
Plots the average retrieval offset per retrieval step from result files.
This works only for results that were generated with `return_err_values = True`
in the :py:class:`core.episodic.EpisodicMemory` object.

Let the last retrieved pattern be *p0*, the associated key be *k* and the retrieval
noise be *n*. Next retrieval is cued with *c = k + n*, which leads to retrival of
*p1*. If *n* is low and *k* is also a pattern in memory: *p1 = k*. Then the retrieval
offset is 0. Otherwise, if *p1 <> k*, the retrieval offset measures how much the distance
from *p0* to *p1* is larger than from *p0* to *k*, so retrieval offset
*v = ||k-p0|| - ||p1-p0||*.

The script can also visualize other properties of episodic memory retrieval,
the respective code can be uncommented and maybe adapted a bit.

Mind that in *error_types* a 0 means that a pattern identical to the key was retrieved,
while for -1 or 1 the pattern was different from the key. In case of 1 this was because
of a retrieval error (within a sequence) and in case of -1 this was because the end of
a sequence was reached.
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# How many errors type 1 (not because of end of sequence)
# Fraction of errors type -1 of all errors
# mean error for type -1
# mean error for type 1
# error sum for type -1
# error sum for type 1
# Fraction of errors type 1 of all possible errors type 1

if __name__=="__main__":

    N = 1
    LOAD = True
    PATHTYPE = "reerror_o18"
    # PATH_PRE = "/run/media/rize/Extreme Festplatte/results/"
    PATH_PRE = "/local/results/"
    # PATH_PRE = "/media/goerlrwh/Extreme Festplatte/results/"

    font = {'family' : 'Sans',
            'size'   : 22}

    matplotlib.rc('font', **font)

    # COLORS = ['k', 'b', 'c', 'g', 'r', 'm', 'k', 'b', 'c', 'g', 'r', 'm']
    COLORS = ['k']*10
    MST = ['d']*10
    NLIST = [0, 1,2,3,4,6]
    MSFACTOR = 1.5
    # MS = list(MSFACTOR*np.array(list(range(len(NLIST)))))
    MS = list(MSFACTOR*np.array(NLIST))


    xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
    NOISES = ["N0", "N1", "N2", "N3", "N4", "N6"]
    # NOISES = ["N0", "N1", "N2", "N3", "N4"]

    letters = "abcdefghijklmnopqrstuvqxyz"[:N]

    try:
        if not LOAD:
            raise Exception()
        err_count = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_count.npy")
        err_mean = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_mean.npy")
        err_sum = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_sum.npy")
        err_count1 = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_count1.npy")
        err_frac = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_frac.npy")
        err_mean1 = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_mean1.npy")
        err_meanm1 = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_meanm1.npy")
        err_sum1 = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_sum1.npy")
        err_summ1 = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_summ1.npy")
        err_made1 = np.load(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_made1.npy")
    except:
        err_count, err_mean, err_sum, err_count1, err_frac, err_mean1, err_meanm1, err_sum1, err_summ1, err_made1 = [], [], [], [], [], [] ,[], [], [], []

        for ni, nn in enumerate(NOISES):
            err_count.append([])
            err_mean.append([])
            err_sum.append([])
            err_count1.append([])
            err_frac.append([])
            err_mean1.append([])
            err_meanm1.append([])
            err_sum1.append([])
            err_summ1.append([])
            err_made1.append([])
            for i, let in enumerate(letters):
                err_count[ni].append([])
                err_mean[ni].append([])
                err_sum[ni].append([])
                err_count1[ni].append([])
                err_frac[ni].append([])
                err_mean1[ni].append([])
                err_meanm1[ni].append([])
                err_sum1[ni].append([])
                err_summ1[ni].append([])
                err_made1[ni].append([])
                dirname = PATH_PRE + PATHTYPE + "{}{}/".format(nn,let)
                print(dirname + "...")
                for ri in range(31):
                    print(ri)
                    fname = "res{}.p".format(ri)
                    with open(dirname + fname, 'rb') as f:
                        res = pickle.load(f)
                    nnz = np.nonzero(res.error_distances)[0]
                    err_count[ni][i].append(len(nnz))
                    err_sum[ni][i].append(np.sum(res.error_distances))
                    err_mean[ni][i].append(np.mean(res.error_distances[nnz]))
                    where1 = np.where(res.error_types == 1)[0]
                    wherem1 = np.where(res.error_types == -1)[0]
                    where0 = np.where(res.error_types == 0)[0]
                    err_count1[ni][i].append(len(where1))
                    err_frac[ni][i].append(len(wherem1)/len(nnz))
                    err_mean1[ni][i].append(np.mean(res.error_distances[where1]))
                    err_meanm1[ni][i].append(np.mean(res.error_distances[wherem1]))
                    err_sum1[ni][i].append(np.sum(res.error_distances[where1]))
                    err_summ1[ni][i].append(np.sum(res.error_distances[wherem1]))
                    len01 = len(where1)+len(where0)
                    if 0==len(where1):
                        err_made1[ni][i].append(0)
                    else:
                        err_made1[ni][i].append(len(where1)/(len(where1)+len(where0)))

        np.save(PATH_PRE+PATHTYPE+NOISES[0]+letters[0]+"/err_count.npy", err_count)
        np.save(PATH_PRE+PATHTYPE+NOISES[0]+letters[0]+"/err_mean.npy", err_mean)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_sum.npy", err_sum)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_count1.npy", err_count1)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_frac.npy", err_frac)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_mean1.npy", err_mean1)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_meanm1.npy", err_meanm1)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_sum1.npy", err_sum1)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_summ1.npy", err_summ1)
        np.save(PATH_PRE + PATHTYPE + NOISES[0] + letters[0] + "/err_made1.npy", err_made1)

    errc = np.mean(err_count,axis=1)
    errm = np.mean(err_mean,axis=1)
    errs = np.mean(err_sum,axis=1)
    errc1 = np.mean(err_count1,axis=1)
    errf = np.mean(err_frac,axis=1)
    errm1 = np.mean(err_mean1,axis=1)
    errmm1 = np.mean(err_meanm1,axis=1)
    errs1 = np.mean(err_sum1,axis=1)
    errsm1 = np.mean(err_summ1,axis=1)
    errma1 = np.mean(err_made1,axis=1)

    RES_VECTORS = [errc, errm, errs, errc1, errf, errm1, errmm1, errs1, errsm1, errma1]
    RES_TITLES = ["error count", "mean error", "error sum", "error type 1 count", "fraction of error type -1", "error type 1 mean", "error type -1 mean", "error type 1 sum", "error type -1 sum",
                  "error type 1 made of all possible type 1"]

    # cnt = len(RES_VECTORS)
    # cols = int(np.sqrt(cnt))
    # rows = int(cnt / cols) + int(bool(cnt % cols))
    # f , ax = plt.subplots(cols, rows, sharex = True, squeeze=False)
    # for ri in range(len(RES_VECTORS)):
    #     rr = int(ri / cols)
    #     cc = ri % cols
    #     for ni, nn in enumerate(NOISES):
    #         ax[cc,rr].plot(xlbls, RES_VECTORS[ri][ni], color=COLORS[ni], marker=MST[ni], ms=ni*MSFACTOR, label=nn)
    #     ax[cc,rr].set_xscale('log')
    #     ax[cc,rr].set_xticks([2, 10, 100, 600])
    #     ax[cc,rr].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #     ax[cc, rr].set_title(RES_TITLES[ri])
    #     ax[cc, rr].legend()
    # plt.show()

    # RES_VECTORS = [errc, errs, errma1]
    # RES_TITLES = ["error count", "offset sum", "proportion of retrieval\n errors committed"]

    RES_VECTORS = [errs]
    RES_TITLES = ["retrieval offset per step"]
    cols=1
    rows=1
    f , ax = plt.subplots(rows, cols, sharex = True, squeeze=False)
    f.subplots_adjust(bottom=0.15, top = 0.85, right=0.9)
    for ri in range(len(RES_VECTORS)):
        rr = int(ri / cols)
        cc = ri % cols
        hand, lab = [], []
        for ni, nn in enumerate(NOISES):
            lab.append("$\sigma$={}".format(nn[1:]))
            if "proportion" in RES_TITLES[ri]:
                l, = ax[rr, cc].plot(xlbls[1:], RES_VECTORS[ri][ni][1:], color=COLORS[ni], marker=MST[ni], ms=MS[ni], label=lab[-1])
            else:
                l, = ax[rr,cc].plot(xlbls, RES_VECTORS[ri][ni]/30000, color=COLORS[ni], marker=MST[ni], ms=MS[ni], label=lab[-1])
            hand.append(l)
        ax[rr, cc].ticklabel_format(axis='y', style='sci', scilimits=[-3, 3])
        ax[rr,cc].set_xscale('log')
        ax[rr,cc].set_xticks([2, 10, 100, 600])
        ax[rr,cc].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax[rr,cc].set_title(RES_TITLES[ri])
        ax[rr, cc].set_ylabel(RES_TITLES[ri])
        # ax[rr,cc].legend()
    ax[0,cols//2].set_xlabel("length of training sequences")

    top = 0.88
    left = 0.1
    mid = 0.375
    right = 0.65

    col = '0.4'
    siz = 30

    # f.text(left, top, "A", color=col, fontsize=siz)
    # f.text(mid, top, "B", color=col, fontsize=siz)
    # f.text(right, top, "C", color=col, fontsize=siz)

    f.legend(hand, lab, 5)
    f.subplots_adjust(left=0.15, right=0.74)
    plt.show()
