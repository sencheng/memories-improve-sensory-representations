"""
(Figure 11)

Loads values extracted by :py:mod:`analReorderNew` from files for different noise levels and
plots feature quality measures against snippet lengths, one line for every noise level.

"""

import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from core import semantic, streamlined
import scipy.stats

if __name__=="__main__":

    N = 16     #irrelevant
    LOAD = True
    PATHTYPE = "reorder_o18"   # Path base, noises and the first letter are appended for the actual paths
    # PATH_PRE = "/run/media/rize/Extreme Festplatte/results/"
    PATH_PRE = "/local/results/"   # Path prefix - actual path to load from is PATH_PRE + PATHTYPE + [noise] + a
    # PATH_PRE = "/media/goerlrwh/Extreme Festplatte/results/"
    STDS = False

    xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
    # NOISES = ["0", "4", "N1", "N2", "N3", "N4", "N5"]
    # LST = ['-', '--', '-.', ':', '-', '--', '-.']
    NOISES = ["N0", "N1", "N2", "N3", "N4", "N6"]    # folder postfixes for different noise levels
    NLIST = [0, 1,2,3,4,6]
    MSFACTOR = 1.5
    # MS = list(MSFACTOR*np.array(list(range(len(NLIST)))))
    MS = list(MSFACTOR*np.array(NLIST))
    # LST = ['-', '--', '-.', ':', '-']
    LST = ['-']*8
    # MST = ['o', 'x', '^', '+', 'd']
    MST = ['d']*8
    SIMPLE_DRAWS = [4]

    NSYM1 = 0
    SYM1 = 'v'
    SYMCOLOR1 = '0.5'
    SYMSIZE1 = 22
    NSYM2 = 4
    SYM2 = '^'
    SYMCOLOR2 = '0.5'
    SYMSIZE2 = 22

    mfcolor = 'none'

    xys_color = 'k'
    xye_color = 'k'
    cats_color = 'k'
    cate_color = 'k'

    letters = "abcdefghijklmnopqrstuvqxyz"[:N]    #only first is used here, because that is where the files were stored in by analReorderNew.py

    # matplotlib.rcParams['lines.linewidth'] = 2
    # matplotlib.rcParams['lines.markersize'] = 12
    # matplotlib.rcParams['lines.markeredgewidth'] = 2
    font = {'family' : 'sans',
            'size'   : 22}

    matplotlib.rc('font', **font)

    d_s, d_e, cat_s, cat_e, rcat_s, rcat_e, xy_s, xy_e, d_lat_s, d_cat_s, d_lat_e, d_cat_e = [], [], [], [], [], [], [], [], [], [], [], []

    fxy, axxy = plt.subplots(3,1,sharex=True)
    fcat, axcat = plt.subplots(3,1,sharex=True)

    l1, l2, lc1r, lc2r, ldl1, ldc1, ldl2, ldc2 = [], [], [], [], [], [], [], []

    for ni, nn in enumerate(NOISES):
        d_s.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_d_s.npy"))
        d_e.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_d_e.npy"))
        cat_s.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_cat_s.npy"))
        cat_e.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_cat_e.npy"))
        rcat_s.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_rcat_s.npy"))
        rcat_e.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_rcat_e.npy"))
        xy_s.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_xy_s.npy"))
        xy_e.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_xy_e.npy"))
        d_lat_s.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_d_lat_s.npy"))
        d_cat_s.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_d_cat_s.npy"))
        d_lat_e.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_d_lat_e.npy"))
        d_cat_e.append(np.load(PATH_PRE+PATHTYPE+nn+letters[0]+"/reorder_d_cat_e.npy"))

        dmean_s = np.mean(d_s[ni], axis=0)
        dmean_e = np.mean(d_e[ni], axis=0)
        dstd_s = np.std(d_s[ni], axis=0)
        dstd_e = np.std(d_e[ni], axis=0)

        catmean_s = np.mean(cat_s[ni], axis=0)
        catmean_e = np.mean(cat_e[ni], axis=0)
        catstd_s = np.std(cat_s[ni], axis=0)
        catstd_e = np.std(cat_e[ni], axis=0)

        rcatmean_s = np.mean(rcat_s[ni], axis=0)
        rcatmean_e = np.mean(rcat_e[ni], axis=0)
        rcatstd_s = np.std(rcat_s[ni], axis=0)
        rcatstd_e = np.std(rcat_e[ni], axis=0)

        xymean_s = np.mean(xy_s[ni], axis=0)
        xymean_e = np.mean(xy_e[ni], axis=0)
        xystd_s = np.std(xy_s[ni], axis=0)
        xystd_e = np.std(xy_e[ni], axis=0)

        d_latmean_s = np.mean(d_lat_s[ni], axis=0)
        d_latmean_e = np.mean(d_lat_e[ni], axis=0)
        d_latstd_s = np.std(d_lat_s[ni], axis=0)
        d_latstd_e = np.std(d_lat_e[ni], axis=0)

        d_catmean_s = np.mean(d_cat_s[ni], axis=0)
        d_catmean_e = np.mean(d_cat_e[ni], axis=0)
        d_catstd_s = np.std(d_cat_s[ni], axis=0)
        d_catstd_e = np.std(d_cat_e[ni], axis=0)

        if ni in SIMPLE_DRAWS:
            axxy[0].plot(xlbls, dmean_s, label="simple", c=xys_color, ls='--')
            axcat[0].plot(xlbls, dmean_s, label="simple", c=cats_color, ls='--')
        axxy[0].plot(xlbls, dmean_e, label="episodic", c=xye_color, ls=LST[ni], marker=MST[ni], ms=MS[ni], mfc=None, clip_on=False)
        axcat[0].plot(xlbls, dmean_e, label="episodic", c=cate_color, ls=LST[ni], marker=MST[ni], ms=MS[ni], mfc=None, clip_on=False)
        if ni == NSYM1:
            axxy[0].plot(xlbls[0], dmean_e[0], c=SYMCOLOR1, mfc=mfcolor, mew=2, marker=SYM1, markersize=SYMSIZE1, zorder=30, clip_on=False)
            axcat[0].plot(xlbls[0], dmean_e[0], c=SYMCOLOR1, mfc=mfcolor, mew=2, marker=SYM1, markersize=SYMSIZE1, zorder=30, clip_on=False)
        if ni == NSYM2:
            axxy[0].plot(xlbls[0], dmean_e[0], c=SYMCOLOR2, mfc=mfcolor, mew=2, marker=SYM2, markersize=SYMSIZE2, zorder=30, clip_on=False)
            axcat[0].plot(xlbls[0], dmean_e[0], c=SYMCOLOR2, mfc=mfcolor, mew=2, marker=SYM2, markersize=SYMSIZE2, zorder=30, clip_on=False)
        if STDS:
            if ni in SIMPLE_DRAWS:
                axxy[0].fill_between(xlbls, dmean_s - dstd_s, dmean_s + dstd_s, facecolor='g', alpha=0.1)
                axcat[0].fill_between(xlbls, dmean_s - dstd_s, dmean_s + dstd_s, facecolor='g', alpha=0.1)
            axxy[0].fill_between(xlbls, dmean_e - dstd_e, dmean_e + dstd_e, facecolor='r', alpha=0.1)
            axcat[0].fill_between(xlbls, dmean_e - dstd_e, dmean_e + dstd_e, facecolor='r', alpha=0.1)

        # lc1, = ax[1].plot(xlbls, catmean_s, c='g', ls='--')
        # lc2, = ax[1].plot(xlbls, catmean_e, c='r', ls='--')
        # ax[1].fill_between(xlbls, catmean_s-catstd_s, catmean_s+catstd_s, facecolor='g', alpha=0.1)
        # ax[1].fill_between(xlbls, catmean_e-catstd_e, catmean_e+catstd_e, facecolor='r', alpha=0.1)
        if ni in SIMPLE_DRAWS:
            tmp, = axcat[1].plot(xlbls, rcatmean_s, c=cats_color, ls='--')
            lc1r.append(tmp)
        tmp, = axcat[1].plot(xlbls, rcatmean_e, c=cate_color, ls=LST[ni], marker=MST[ni], ms=MS[ni], mfc=None, clip_on=False)
        lc2r.append(tmp)
        if STDS:
            if ni in SIMPLE_DRAWS:
                axcat[1].fill_between(xlbls, rcatmean_s - rcatstd_s, rcatmean_s + rcatstd_s, facecolor='y', alpha=0.1)
            axcat[1].fill_between(xlbls, rcatmean_e - rcatstd_e, rcatmean_e + rcatstd_e, facecolor='m', alpha=0.1)
        if ni in SIMPLE_DRAWS:
            tmp, = axxy[1].plot(xlbls, xymean_s, label="simple", c=xys_color, ls='--')
            l1.append(tmp)
        tmp, = axxy[1].plot(xlbls, xymean_e, label="episodic", c=xye_color, ls=LST[ni], marker=MST[ni], ms=MS[ni], mfc=None, clip_on=False)
        l2.append(tmp)
        # if ni == NSYM1:
        #     axxy[1].plot(xlbls[0], xymean_e[0], c=SYMCOLOR1, mfc=mfcolor, mew=2, marker=SYM1, markersize=SYMSIZE1, zorder=30, clip_on=False)
        #     axcat[1].plot(xlbls[0], rcatmean_e[0], c=SYMCOLOR1, mfc=mfcolor, mew=2, marker=SYM1, markersize=SYMSIZE1, zorder=30, clip_on=False)
        # if ni == NSYM2:
        #     axxy[1].plot(xlbls[0], xymean_e[0], c=SYMCOLOR2, mfc=mfcolor, mew=2, marker=SYM2, markersize=SYMSIZE2, zorder=30, clip_on=False)
        #     axcat[1].plot(xlbls[0], rcatmean_e[0], c=SYMCOLOR2, mfc=mfcolor, mew=2, marker=SYM2, markersize=SYMSIZE2, zorder=30, clip_on=False)
        if STDS:
            if ni in SIMPLE_DRAWS:
                axxy[1].fill_between(xlbls, xymean_s - xystd_s, xymean_s + xystd_s, facecolor='g', alpha=0.1)
            axxy[1].fill_between(xlbls, xymean_e - xystd_e, xymean_e + xystd_e, facecolor='r', alpha=0.1)
        if ni in SIMPLE_DRAWS:
            tmp, = axxy[2].plot(xlbls, d_latmean_s, c=xys_color, ls='--')
            ldl1.append(tmp)
            tmp, = axcat[2].plot(xlbls, d_catmean_s, c=cats_color, ls='--')
            ldc1.append(tmp)
        if STDS:
            if ni in SIMPLE_DRAWS:
                axxy[2].fill_between(xlbls, d_latmean_s - d_latstd_s, d_latmean_s + d_latstd_s, facecolor='g', alpha=0.1)
                axcat[2].fill_between(xlbls, d_catmean_s - d_catstd_s, d_catmean_s - d_catstd_s, facecolor='y', alpha=0.1)
        tmp, = axxy[2].plot(xlbls, d_latmean_e, c=xye_color, ls=LST[ni], marker=MST[ni], ms=MS[ni], mfc=None, clip_on=False)
        ldl2.append(tmp)
        tmp, = axcat[2].plot(xlbls, d_catmean_e, c=cate_color, ls=LST[ni], marker=MST[ni], ms=MS[ni], mfc=None, clip_on=False)
        ldc2.append(tmp)
        # if ni == NSYM1:
            # axxy[2].plot(xlbls[0], d_latmean_e[0], c=SYMCOLOR1, mfc=mfcolor, mew=2, marker=SYM1, markersize=SYMSIZE1, zorder=30, clip_on=False)
            # axcat[2].plot(xlbls[0], d_catmean_e[0], c=SYMCOLOR1, mfc=mfcolor, mew=2, marker=SYM1, markersize=SYMSIZE1, zorder=30, clip_on=False)
        # if ni == NSYM2:
            # axxy[2].plot(xlbls[0], d_latmean_e[0], c=SYMCOLOR2, mfc=mfcolor, mew=2, marker=SYM2, markersize=SYMSIZE2, zorder=30, clip_on=False)
            # axcat[2].plot(xlbls[0], d_catmean_e[0], c=SYMCOLOR2, mfc=mfcolor, mew=2, marker=SYM2, markersize=SYMSIZE2, zorder=30, clip_on=False)
        if STDS:
            axxy[2].fill_between(xlbls, d_latmean_e - d_latstd_e, d_latmean_e - d_latstd_e, facecolor='r', alpha=0.1)
            axcat[2].fill_between(xlbls, d_catmean_e - d_catstd_e, d_catmean_e - d_catstd_e, facecolor='m', alpha=0.1)

    axxy[0].set_xscale('log')
    axcat[0].set_xscale('log')
    axxy[0].set_xticks([2, 10, 100, 600])
    axcat[0].set_xticks([2, 10, 100, 600])
    axxy[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axcat[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axxy[0].set_ylabel("delta value\nof SFA output")
    axcat[0].set_ylabel("delta value\nof SFA output")
    # axxy[0].set_title("slowness of 3 slowest features")
    # axcat[0].set_title("slowness of 3 slowest features")
    # ax[0].legend()
    axxy[1].set_ylabel('r of variable\nprediction')
    axcat[1].set_ylabel('r of variable\nprediction')
    # axxy[1].set_title('correlation of features with x/y')
    # axcat[1].set_title('correlation of features with category')
    # axxy[2].set_title('delta value of x/y')
    # axcat[2].set_title('delta value of category')
    # axxy[1].legend((l1[-1], lc1r[-1], l2[0], lc2r[0]), ("x/y", "category", "x/y", "category"), loc=4)
    # axcat[1].legend((l1[-1], lc1r[-1], l2[0], lc2r[0]), ("x/y", "category", "x/y", "category"), loc=4)
    # ax[1].set_xlabel("length of forming sequences")
        # f.subplots_adjust(bottom=0.15)

    axxy[2].set_ylabel("delta value\n of x,y")
    axxy[2].set_xlabel("length of training episodes")
    axcat[2].set_ylabel("delta value\n of obj. identity")
    axcat[2].set_xlabel("length of training episodes")

    left = 0.1
    top = 0.95
    mid = 0.68
    bot = 0.41
    col = '0.4'
    siz = 30

    fxy.text(left, top, "A", color=col, fontsize=siz)
    fxy.text(left, mid, "B", color=col, fontsize=siz)
    fxy.text(left, bot, "C", color=col, fontsize=siz)

    fxy.legend((l1[-1],l2[0]), ("simple", "episodic"), 4)
    fcat.legend((lc1r[-1], lc2r[0]), ("simple", "episodic"), 4)
    # plt.figlegend((l1[0],l1[1],l1[2],l1[3],l1[4],l1[5],l1[6]), ("n=0", "n=0.2", "n=1", "n=2", "n=3", "n=4", "n=5"), 2)
    fxy.legend([l2[i] for i in range(len(NLIST))], ["$\sigma$={}".format(i) for i in NLIST], 5)
    fcat.legend([ldc2[i] for i in range(len(NLIST))], ["$\sigma$={}".format(i) for i in NLIST], 5)
    # plt.figlegend((lc1,l1), ("object category", "x/r regression r²"), 2)

    fxy.subplots_adjust(top=0.95, bottom=0.18, right=0.87)
    fcat.subplots_adjust(top=0.95, bottom=0.18, right=0.87)
    plt.close(fcat)
    plt.show()
