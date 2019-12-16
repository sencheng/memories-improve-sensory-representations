"""
(Figures 10, 11)

Loads result file batches to average over and plots feature quality measures against snippet length.
Before plotting, stores the values extracted from the results files in the first of the results folders,
to speed up consecutive script executions and to enable loading the values for different noise levels
in order to make figure 11 using :py:mod:`analReorderNoise`.

"""

import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from core import semantic, streamlined
import scipy.stats
import sklearn.linear_model

if __name__=="__main__":

    N = 16         # How many grid runs with the same parameters to load and average over
    LOAD = True
    PATHTYPE = "reorder_o18"    # Path base, letters are appended for the actual paths
    # PATH_PRE = "/run/media/rize/Extreme Festplatte/results/"
    PATH_PRE = "/local/results/"  # Path prefix - actual path to load from is PATH_PRE + PATHTYPE + [letter]
    # PATH_PRE = "/media/goerlrwh/Extreme Festplatte/results/"

    xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
    # xlbls = [2,3,5,8,12,20,40,100,200,600]
                    # x axis is snippet length, these are the possible values, must fit to results

    letters = "abcdefghijklmnopqrstuvqxyz"[:N]
    # letters = "n"[:N]

    dcolors = ['k', 'k']
    dstyles = ['--', '-']
    ccolors = ['k', '0.6', 'k', '0.6']
    cstyles = ['--', '--', '-', '-']


    MS = 22
    MEW = 2
    font = {'family' : 'Sans',
            'size'   : 22}

    matplotlib.rc('font', **font)

    NFEAT = 3

    # Whitening settings
    # of the data
    WHITENER = True
    NORM = False
    # for analysis
    CORRNORM = False
    # DELTANORM = True

    TRAIN_LEARNER = True

    try:
        if not LOAD:
            raise Exception()
        d_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_s.npy")
        d_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_e.npy")
        cat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_cat_s.npy")
        cat_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_cat_e.npy")
        rcat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_s.npy")
        rcat_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_e.npy")
        xy_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_s.npy")
        xy_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_e.npy")
        phiZ_S = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_phiZ_S.npy")
        phiZ_E = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_phiZ_E.npy")
        d_lat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_s.npy")
        d_cat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_s.npy")
        d_lat_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_e.npy")
        d_cat_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_e.npy")
        d_form = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_d_form.npy")
        d_retrieved = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_d_retrieved.npy")
    except:
        d_s, d_e, cat_s, cat_e, rcat_s, rcat_e, xy_s, xy_e, d_lat_s, d_cat_s, d_lat_e, d_cat_e = [], [], [], [], [], [], [], [], [], [], [], []
        phiZ_S, phiZ_E, d_form, d_retrieved = [], [], [], []
        for i, let in enumerate(letters):
            dirname = PATH_PRE+PATHTYPE+"{}/".format(let)
            print(dirname+"...")
            testing_data = np.load(dirname + "testing0.npz")
            seq2, cat2, lat2, _ = testing_data['testing_sequenceX'], testing_data['testing_categories'], testing_data['testing_latent'], testing_data['testing_ranges']
            d_s.append([])
            d_e.append([])
            cat_s.append([])
            cat_e.append([])
            rcat_s.append([])
            rcat_e.append([])
            xy_s.append([])
            xy_e.append([])
            d_lat_s.append([])
            d_cat_s.append([])
            d_lat_e.append([])
            d_cat_e.append([])
            phiZ_S.append([])
            phiZ_E.append([])
            d_form.append([])
            d_retrieved.append([])
            for ri in range(len(xlbls)):
                print(ri)
                fname = "res{}.p".format(ri)
                with open(dirname + fname, 'rb') as f:
                    res = pickle.load(f)
                forming_data = np.load(dirname + "forming{}.npz".format(ri))
                formseq, formcat, formlat = forming_data['forming_sequenceX'], forming_data['forming_categories'], forming_data['forming_latent']
                d_s[i].append(np.mean(np.sort(res.d_values['testingZ_S'])[:NFEAT]))
                d_e[i].append(np.mean(np.sort(res.d_values['testingZ_E'])[:NFEAT]))

                d_lat_s[i].append(np.mean(res.d_values['forming_lat'][:2]))
                d_cat_s[i].append(np.mean(res.d_values['forming_cat']))
                d_lat_e[i].append(np.mean(res.d_values['retrieved_lat'][:2]))
                d_cat_e[i].append(np.mean(res.d_values['retrieved_cat']))

                d_form[i].append(np.mean(np.sort(res.d_values['forming_Y'])[:16]))
                d_retrieved[i].append(np.mean(np.sort(res.d_values['retrieved_Y'])[:16]))

                cat_s[i].append(np.max(np.abs(res.testing_corrZ_simple[4, :])))
                cat_e[i].append(np.max(np.abs(res.testing_corrZ_episodic[4, :])))

                sfa1 = semantic.load_SFA(dirname + res.data_description + "train0.sfa")
                yy2 = semantic.exec_SFA(sfa1, seq2)
                if WHITENER:
                    yy2_w = res.whitener(yy2)
                elif NORM:
                    yy2_w = streamlined.normalizer(yy2, res.params.normalization)(yy2)
                if WHITENER or NORM:
                    zz2S = semantic.exec_SFA(res.sfa2S, yy2_w)
                else:
                    zz2S = semantic.exec_SFA(res.sfa2S, yy2)
                zz2S_w = streamlined.normalizer(zz2S, res.params.normalization)(zz2S)
                if WHITENER or NORM:
                    zz2E = semantic.exec_SFA(res.sfa2E, yy2_w)
                else:
                    zz2E = semantic.exec_SFA(res.sfa2E, yy2)
                zz2E_w = streamlined.normalizer(zz2E, res.params.normalization)(zz2E)

                formseqY = semantic.exec_SFA(sfa1, formseq)
                if WHITENER:
                    formseqY_w = res.whitener(formseqY)
                elif NORM:
                    formseqY_w = streamlined.normalizer(formseqY, res.params.normalization)(formseqY)

                target_matrixS = np.append(formlat, formcat[:, None], axis=1)
                if CORRNORM:
                    training_matrixS = semantic.exec_SFA(res.sfa2S, formseqY_w)
                else:
                    training_matrixS = semantic.exec_SFA(res.sfa2S, formseqY)
                learnerS = sklearn.linear_model.LinearRegression()
                learnerS.fit(training_matrixS[:,:NFEAT], target_matrixS)

                if CORRNORM:
                    training_matrixE = semantic.exec_SFA(res.sfa2E, formseqY_w)    # this is actually not accurate, i would need to use retrieved sequence, which I don't have here
                                                                                # also, I would need to use retrieved lat and cat, which I don't have either
                else:
                    training_matrixE = semantic.exec_SFA(res.sfa2E, formseqY)
                learnerE = sklearn.linear_model.LinearRegression()
                learnerE.fit(training_matrixE[:,:NFEAT], target_matrixS)

                if not TRAIN_LEARNER:
                    if CORRNORM:
                        predictionS = res.learnerS.predict(zz2S_w)
                        predictionE = res.learnerE.predict(zz2E_w)
                    else:
                        predictionS = res.learnerS.predict(zz2S)
                        predictionE = res.learnerE.predict(zz2E)
                else:
                    if CORRNORM:
                        predictionS = learnerS.predict(zz2S_w[:,:NFEAT])
                        predictionE = learnerE.predict(zz2E_w[:,:NFEAT])
                    else:
                        predictionS = learnerS.predict(zz2S_w[:, :NFEAT])
                        predictionE = learnerE.predict(zz2E_w[:, :NFEAT])
                _, _, r_valueX_S, _, _ = scipy.stats.linregress(lat2[:, 0], predictionS[:, 0])
                _, _, r_valueY_S, _, _ = scipy.stats.linregress(lat2[:, 1], predictionS[:, 1])
                _, _, r_valueCAT_S, _, _ = scipy.stats.linregress(cat2, predictionS[:, 4])
                _, _, r_valueCosphi_S, _, _ = scipy.stats.linregress(lat2[:, 2], predictionS[:, 2])
                _, _, r_valueSinphi_S, _, _ = scipy.stats.linregress(lat2[:, 3], predictionS[:, 3])
                _, _, r_valueX_E, _, _ = scipy.stats.linregress(lat2[:, 0], predictionE[:, 0])
                _, _, r_valueY_E, _, _ = scipy.stats.linregress(lat2[:, 1], predictionE[:, 1])
                _, _, r_valueCAT_E, _, _ = scipy.stats.linregress(cat2, predictionE[:, 4])
                _, _, r_valueCosphi_E, _, _ = scipy.stats.linregress(lat2[:, 2], predictionE[:, 2])
                _, _, r_valueSinphi_E, _, _ = scipy.stats.linregress(lat2[:, 3], predictionE[:, 3])

                xy_s[i].append(np.mean((r_valueX_S, r_valueY_S)))
                rcat_s[i].append(r_valueCAT_S)
                phiZ_S[i].append(np.mean((r_valueCosphi_S, r_valueSinphi_S)))
                xy_e[i].append(np.mean((r_valueX_E, r_valueY_E)))
                rcat_e[i].append(r_valueCAT_E)
                phiZ_E[i].append(np.mean((r_valueCosphi_E, r_valueSinphi_E)))

        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_s.npy", d_s)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_e.npy", d_e)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_cat_s.npy", cat_s)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_cat_e.npy", cat_e)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_s.npy", rcat_s)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_e.npy", rcat_e)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_s.npy", xy_s)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_e.npy", xy_e)
        np.save(PATH_PRE + PATHTYPE + letters[0] + "/reorder_phiZ_S.npy", phiZ_S)
        np.save(PATH_PRE + PATHTYPE + letters[0] + "/reorder_phiZ_E.npy", phiZ_E)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_s.npy", d_lat_s)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_s.npy", d_cat_s)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_e.npy", d_lat_e)
        np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_e.npy", d_cat_e)
        np.save(PATH_PRE + PATHTYPE + letters[0] + "/reorder_d_form.npy", d_form)
        np.save(PATH_PRE + PATHTYPE + letters[0] + "/reorder_d_retrieved.npy", d_retrieved)

    dmean_s = np.mean(d_s,axis=0)
    dmean_e = np.mean(d_e,axis=0)
    dstd_s = np.std(d_s,axis=0)
    dstd_e = np.std(d_e,axis=0)

    catmean_s = np.mean(cat_s,axis=0)
    catmean_e = np.mean(cat_e,axis=0)
    catstd_s = np.std(cat_s,axis=0)
    catstd_e = np.std(cat_e,axis=0)

    rcatmean_s = np.mean(rcat_s,axis=0)
    rcatmean_e = np.mean(rcat_e,axis=0)
    rcatstd_s = np.std(rcat_s,axis=0)
    rcatstd_e = np.std(rcat_e,axis=0)

    xymean_s = np.mean(xy_s,axis=0)
    xymean_e = np.mean(xy_e,axis=0)
    xystd_s = np.std(xy_s,axis=0)
    xystd_e = np.std(xy_e,axis=0)

    phimean_s = np.mean(phiZ_S,axis=0)
    phimean_e = np.mean(phiZ_E,axis=0)
    phistd_s = np.std(phiZ_S,axis=0)
    phistd_e = np.std(phiZ_E,axis=0)

    d_latmean_s = np.mean(d_lat_s,axis=0)
    d_latmean_e = np.mean(d_lat_e,axis=0)
    d_latstd_s = np.std(d_lat_s,axis=0)
    d_latstd_e = np.std(d_lat_e,axis=0)

    d_catmean_s = np.mean(d_cat_s,axis=0)
    d_catmean_e = np.mean(d_cat_e,axis=0)
    d_catstd_s = np.std(d_cat_s,axis=0)
    d_catstd_e = np.std(d_cat_e,axis=0)

    d_formmean = np.mean(d_form,axis=0)
    d_retrievedmean = np.mean(d_retrieved,axis=0)
    d_formstd = np.std(d_form,axis=0)
    d_retrievedstd = np.std(d_retrieved,axis=0)

    f, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(xlbls, dmean_s, label="simple", c=dcolors[0], ls=dstyles[0])
    ax[0].plot(xlbls, dmean_e, label="episodic", c=dcolors[1], ls=dstyles[1])
    # ax[0].fill_between(xlbls, dmean_s-dstd_s, dmean_s+dstd_s, facecolor='g', alpha=0.1)
    # ax[0].fill_between(xlbls, dmean_e-dstd_e, dmean_e+dstd_e, facecolor='r', alpha=0.1)
    ax[0].set_xscale('log')
    ax[0].set_xticks([2, 10, 100, 600])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax[0].set_xlabel("Length of training episodes")
    ax[0].set_ylabel("delta value\nof SFA output")
    # ax[0].legend()
    ax[0].plot(xlbls[0], dmean_s[0], c="k", mfc="none", marker="^", markersize=MS, mew=MEW, clip_on=False)
    ax[0].plot(xlbls[0], dmean_e[0], c="k", mfc="none", marker="v", markersize=MS, mew=MEW, clip_on=False)
    # ax[0].plot(xlbls[0], dmean_s[0], c="b", mfc="none", marker="+", markersize=MS)
    # ax[0].plot(xlbls[25], dmean_s[25], c="b", mfc="none", marker="x", markersize=MS)

    # lc1, = ax[1].plot(xlbls, catmean_s, c='g', ls='--')
    # lc2, = ax[1].plot(xlbls, catmean_e, c='r', ls='--')
    # ax[1].fill_between(xlbls, catmean_s-catstd_s, catmean_s+catstd_s, facecolor='g', alpha=0.1)
    # ax[1].fill_between(xlbls, catmean_e-catstd_e, catmean_e+catstd_e, facecolor='r', alpha=0.1)
    lc1r, = ax[1].plot(xlbls, rcatmean_s, c=ccolors[1], ls=cstyles[1])
    lc2r, = ax[1].plot(xlbls, rcatmean_e, c=ccolors[3], ls=cstyles[3])
    # ax[1].fill_between(xlbls, rcatmean_s-rcatstd_s, rcatmean_s+rcatstd_s, facecolor='g', alpha=0.1)
    # ax[1].fill_between(xlbls, rcatmean_e-rcatstd_e, rcatmean_e+rcatstd_e, facecolor='r', alpha=0.1)
    l1, = ax[1].plot(xlbls, xymean_s, label="simple", c=ccolors[0], ls=cstyles[0])
    l2, = ax[1].plot(xlbls, xymean_e, label="episodic", c=ccolors[2], ls=cstyles[2])
    # ax[1].fill_between(xlbls, xymean_s-xystd_s, xymean_s+xystd_s, facecolor='g', alpha=0.1)
    # ax[1].fill_between(xlbls, xymean_e-xystd_e, xymean_e+xystd_e, facecolor='r', alpha=0.1)
    # lp1, = ax[1].plot(xlbls, phimean_s, label="simple", c='g', ls=':')
    # lp2, = ax[1].plot(xlbls, phimean_e, label="episodic", c='r', ls=':')
    ax[1].set_xscale('log')
    ax[1].set_xticks([2, 10, 100, 600])
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].set_ylabel("r of variable\nprediction")
    # ax[1].plot(xlbls[0], catmean_s[0], c="k", mfc="none", marker="o", markersize=MS)
    # ax[1].plot(xlbls[0], catmean_e[0], c="k", mfc="none", marker="s", markersize=MS)
    # ax[1].plot(xlbls[0], xymean_s[0], c="k", mfc="none", marker="o", markersize=MS, mew=MEW, clip_on=False)
    # ax[1].plot(xlbls[0], xymean_e[0], c="k", mfc="none", marker="s", markersize=MS, mew=MEW, clip_on=False)
    # ax[1].plot(xlbls[0], rcatmean_s[0], c="k", mfc="none", marker="o", markersize=MS, mew=MEW, clip_on=False)
    # ax[1].plot(xlbls[0], rcatmean_e[0], c="k", mfc="none", marker="s", markersize=MS, mew=MEW, clip_on=False)
    # ax[1].plot(xlbls[0], catmean_s[0], c="b", mfc="none", marker="+", markersize=MS)
    # ax[1].plot(xlbls[25], catmean_s[25], c="b", mfc="none", marker="x", markersize=MS)
    # ax[1].plot(xlbls[0], xymean_s[0], c="b", mfc="none", marker="+", markersize=MS)
    # ax[1].plot(xlbls[25], xymean_s[25], c="b", mfc="none", marker="x", markersize=MS)
    # ax[1].plot(xlbls[0], rcatmean_s[0], c="b", mfc="none", marker="+", markersize=MS)
    # ax[1].plot(xlbls[25], rcatmean_s[25], c="b", mfc="none", marker="x", markersize=MS)
    # ax[1].legend((lc1, lc1r,l1), ("object category", "cat regression r²", "x/r regression r²"), loc=4)
    # ax[1].legend((l1, lp1, lc1r), ("x/y", "cos/sin(phi)", "category"), loc=4)
    # ax[1].legend((l2, lc2r), ("x/y", "identity"), loc=4)
    # f.subplots_adjust(bottom=0.15)

    ax[2].plot(xlbls, d_latmean_s, c=ccolors[0], ls=cstyles[0])
    ax[2].plot(xlbls, d_catmean_s, c=ccolors[1], ls=cstyles[1])
    # ax[2].fill_between(xlbls, d_latmean_s-d_latstd_s, d_latmean_s+d_latstd_s, facecolor='g', alpha=0.1)
    # ax[2].fill_between(xlbls, d_catmean_s-d_catstd_s, d_catmean_s-d_catstd_s, facecolor='g', alpha=0.1)
    # ax[2].plot(xlbls[0], d_catmean_s[0], c="k", mfc="none", marker="o", markersize=MS, mew=MEW, clip_on=False)
    # ax[2].plot(xlbls[0], d_catmean_e[0], c="k", mfc="none", marker="s", markersize=MS, mew=MEW, clip_on=False)
    # ax[2].plot(xlbls[0], d_latmean_s[0], c="k", mfc="none", marker="o", markersize=MS, mew=MEW, clip_on=False)
    # ax[2].plot(xlbls[0], d_latmean_e[0], c="k", mfc="none", marker="s", markersize=MS, mew=MEW, clip_on=False)
    ldl, = ax[2].plot(xlbls, d_latmean_e, c=ccolors[2], ls=cstyles[2])
    ldc, = ax[2].plot(xlbls, d_catmean_e, c=ccolors[3], ls=cstyles[3])

    # ldf, = ax[2].plot(xlbls, d_formmean, c='g', ls = ':')
    # ax[2].plot(xlbls, d_retrievedmean, c='r', ls = ':')
    # ax[2].fill_between(xlbls, d_latmean_e-d_latstd_e, d_latmean_e-d_latstd_e, facecolor='r', alpha=0.1)
    # ax[2].fill_between(xlbls, d_catmean_e-d_catstd_e, d_catmean_e-d_catstd_e, facecolor='r', alpha=0.1)
    # ax[2].plot(xlbls[0], d_catmean_s[0], c="b", mfc="none", marker="+", markersize=MS)
    # ax[2].plot(xlbls[25], d_catmean_s[25], c="b", mfc="none", marker="x", markersize=MS)
    # ax[2].plot(xlbls[0], d_latmean_s[0], c="b", mfc="none", marker="+", markersize=MS)
    # ax[2].plot(xlbls[25], d_latmean_s[25], c="b", mfc="none", marker="x", markersize=MS)
    # ax[2].legend((ldl, ldc, ldf), ("x/y", "category", "form/retr_y"))
    # ax[2].legend((ldl, ldc), ("x/y", "identity"))
    ax[2].set_ylabel("delta value\nof latents")
    ax[2].set_xlabel("length of training episodes")

    left = 0.095
    top = 0.95
    mid = 0.695
    bot = 0.43
    col = '0.4'
    siz = 30

    f.text(left, top, "A", color=col, fontsize=siz)
    f.text(left, mid, "B", color=col, fontsize=siz)
    f.text(left, bot, "C", color=col, fontsize=siz)

    plt.figlegend((ldl, ldc,l1,l2), ("x,y", "identity", "simple", "episodic"), 4, ncol=2)
    # plt.figlegend((ldl, ldc), ("x,y", "identity"), 8)
    plt.subplots_adjust(top=0.95, bottom=0.22)
    # plt.figlegend((lc1,l1), ("object category", "x/r regression r²"), 2)
    plt.show()
