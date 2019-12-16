import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from core import semantic, streamlined, tools
import scipy.stats
import sklearn.linear_model

N = 13
REPS_TOTAL = 10
REPS = 3
LOAD = True
PATH_PRE = "/local/results/"
PATHTYPE = "reorder_replay_"

xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]

letters = "abcdefghijklmnopqrstuvqxyz"[:N]

NFEAT = 3

TRAIN_LEARNER = True

try:
    if not LOAD:
        raise Exception()
    d_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_s.npy")
    d_e_retr = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_e_retr.npy")
    d_e_rep = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_d_e_rep.npy")
    rcat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_s.npy")
    rcat_e_retr = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_e_retr.npy")
    rcat_e_rep = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_rcat_e_rep.npy")
    xy_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_s.npy")
    xy_e_retr = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_e_retr.npy")
    xy_e_rep = np.load(PATH_PRE + PATHTYPE + letters[0] + "/reorder_xy_e_rep.npy")
    d_lat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_s.npy")
    d_cat_s = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_s.npy")
    d_lat_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_e.npy")
    d_cat_e = np.load(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_e.npy")
except:
    d_s, d_e_retr, cat_s, cat_e_retr, rcat_s, rcat_e_retr, xy_s, xy_e_retr, d_lat_s, d_cat_s, d_lat_e, d_cat_e = [], [], [], [], [], [], [], [], [], [], [], []
    d_e_rep, cat_e_rep, rcat_e_rep, xy_e_rep = [], [], [], []
    for i, let in enumerate(letters):
        dirname = PATH_PRE+PATHTYPE+"{}/".format(let)
        print(dirname+"...")
        testing_data = np.load(dirname + "testing0.npz")
        seq2, cat2, lat2, _ = testing_data['testing_sequenceX'], testing_data['testing_categories'], testing_data['testing_latent'], testing_data['testing_ranges']
        with open(dirname + "res0.p", 'rb') as f:
            res = pickle.load(f)
        sfa1 = semantic.load_SFA(dirname + res.data_description + "train0.sfa")
        yy2 = semantic.exec_SFA(sfa1, seq2)
        yy2_w = streamlined.normalizer(yy2, res.params.normalization)(yy2)
        
        d_s.append([])
        d_e_retr.append([])
        d_e_rep.append([])
        rcat_s.append([])
        rcat_e_retr.append([])
        rcat_e_rep.append([])
        xy_s.append([])
        xy_e_retr.append([])
        xy_e_rep.append([])
        d_lat_s.append([])
        d_cat_s.append([])
        d_lat_e.append([])
        d_cat_e.append([])
        for ri in range(len(xlbls)):
            d_s[i].append([])
            d_e_retr[i].append([])
            d_e_rep[i].append([])
            rcat_s[i].append([])
            rcat_e_retr[i].append([])
            rcat_e_rep[i].append([])
            xy_s[i].append([])
            xy_e_retr[i].append([])
            xy_e_rep[i].append([])
            print(ri)
            fname = "res{}.p".format(ri)
            with open(dirname + fname, 'rb') as f:
                res = pickle.load(f)
            forming_data = np.load(dirname + "forming{}.npz".format(ri))
            formseq, formcat, formlat = forming_data['forming_sequenceX'], forming_data['forming_categories'], forming_data['forming_latent']

            formseqY = semantic.exec_SFA(sfa1, formseq)
            formseqY_w = streamlined.normalizer(formseqY, res.params.normalization)(formseqY)

            d_lat_s[i].append(np.mean(res.d_values['forming_lat'][:2]))
            d_cat_s[i].append(np.mean(res.d_values['forming_cat']))
            d_lat_e[i].append(np.mean(res.d_values['retrieved_lat'][:2]))
            d_cat_e[i].append(np.mean(res.d_values['retrieved_cat']))

            for irep in range(REPS_TOTAL):

                sfa2S = semantic.load_SFA(dirname + "sfa2S_res{}_repeat{}.sfa".format(ri, irep))
                sfa2E_retr = semantic.load_SFA(dirname + "sfa2E_res{}_retr{}.sfa".format(ri, irep))
                sfa2E_rep = semantic.load_SFA(dirname + "sfa2E_res{}_repeat{}.sfa".format(ri, irep))

                zz2S = semantic.exec_SFA(sfa2S, yy2_w)
                zz2S_w = streamlined.normalizer(zz2S, res.params.normalization)(zz2S)
                zz2E_retr = semantic.exec_SFA(sfa2E_retr, yy2_w)
                zz2E_retr_w = streamlined.normalizer(zz2E_retr, res.params.normalization)(zz2E_retr)
                zz2E_rep = semantic.exec_SFA(sfa2E_rep, yy2_w)
                zz2E_rep_w = streamlined.normalizer(zz2E_rep, res.params.normalization)(zz2E_rep)

                deltas_S = tools.delta_diff(zz2S_w)
                deltas_E_retr = tools.delta_diff(zz2E_retr_w)
                deltas_E_rep = tools.delta_diff(zz2E_rep_w)

                d_s[i][ri].append(np.mean(deltas_S[:NFEAT]))
                d_e_retr[i][ri].append(np.mean(deltas_E_retr[:NFEAT]))
                d_e_rep[i][ri].append(np.mean(deltas_E_rep[:NFEAT]))

                target_matrixS = np.append(formlat, formcat[:, None], axis=1)
                training_matrixS = semantic.exec_SFA(sfa2S, formseqY_w)
                learnerS = sklearn.linear_model.LinearRegression()
                learnerS.fit(training_matrixS[:,:NFEAT], target_matrixS)

                training_matrixE_retr = semantic.exec_SFA(sfa2E_retr, formseqY_w)    # this is actually not accurate, i would need to use retrieved sequence, which I don't have here
                                                                                        # also, I would need to use retrieved lat and cat, which I don't have either

                training_matrixE_rep = semantic.exec_SFA(sfa2E_rep, formseqY_w)
                
                learnerE_retr = sklearn.linear_model.LinearRegression()
                learnerE_retr.fit(training_matrixE_retr[:,:NFEAT], target_matrixS)
                learnerE_rep = sklearn.linear_model.LinearRegression()
                learnerE_rep.fit(training_matrixE_rep[:, :NFEAT], target_matrixS)

                if not TRAIN_LEARNER:
                    predictionS = res.learnerS.predict(zz2S_w)
                    predictionE_retr = res.learnerE.predict(zz2E_retr_w)
                    predictionE_rep = res.learnerE.predict(zz2E_rep_w)
                else:
                    predictionS = learnerS.predict(zz2S_w[:,:NFEAT])
                    predictionE_retr = learnerE_retr.predict(zz2E_retr_w[:,:NFEAT])
                    predictionE_rep = learnerE_rep.predict(zz2E_rep_w[:, :NFEAT])
                _, _, r_valueX_S, _, _ = scipy.stats.linregress(lat2[:, 0], predictionS[:, 0])
                _, _, r_valueY_S, _, _ = scipy.stats.linregress(lat2[:, 1], predictionS[:, 1])
                _, _, r_valueCAT_S, _, _ = scipy.stats.linregress(cat2, predictionS[:, 4])
                # _, _, r_valueCosphi_S, _, _ = scipy.stats.linregress(lat2[:, 2], predictionS[:, 2])
                # _, _, r_valueSinphi_S, _, _ = scipy.stats.linregress(lat2[:, 3], predictionS[:, 3])
                _, _, r_valueX_E_retr, _, _ = scipy.stats.linregress(lat2[:, 0], predictionE_retr[:, 0])
                _, _, r_valueY_E_retr, _, _ = scipy.stats.linregress(lat2[:, 1], predictionE_retr[:, 1])
                _, _, r_valueCAT_E_retr, _, _ = scipy.stats.linregress(cat2, predictionE_retr[:, 4])
                _, _, r_valueX_E_rep, _, _ = scipy.stats.linregress(lat2[:, 0], predictionE_rep[:, 0])
                _, _, r_valueY_E_rep, _, _ = scipy.stats.linregress(lat2[:, 1], predictionE_rep[:, 1])
                _, _, r_valueCAT_E_rep, _, _ = scipy.stats.linregress(cat2, predictionE_rep[:, 4])
                # _, _, r_valueCosphi_E, _, _ = scipy.stats.linregress(lat2[:, 2], predictionE[:, 2])
                # _, _, r_valueSinphi_E, _, _ = scipy.stats.linregress(lat2[:, 3], predictionE[:, 3])

                xy_s[i][ri].append(np.mean((r_valueX_S, r_valueY_S)))
                rcat_s[i][ri].append(r_valueCAT_S)
                # phiZ_S.append(np.mean((r_valueCosphi_S, r_valueSinphi_S)))
                xy_e_retr[i][ri].append(np.mean((r_valueX_E_retr, r_valueY_E_retr)))
                rcat_e_retr[i][ri].append(r_valueCAT_E_retr)
                xy_e_rep[i][ri].append(np.mean((r_valueX_E_rep, r_valueY_E_rep)))
                rcat_e_rep[i][ri].append(r_valueCAT_E_rep)
                # phiZ_E.append(np.mean((r_valueCosphi_E, r_valueSinphi_E)))

    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_s.npy", d_s)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_e_retr.npy", d_e_retr)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_e_rep.npy", d_e_rep)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_s.npy", rcat_s)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_rcat_e_retr.npy", rcat_e_retr)
    np.save(PATH_PRE + PATHTYPE + letters[0] + "/reorder_rcat_e_rep.npy", rcat_e_rep)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_s.npy", xy_s)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_xy_e_retr.npy", xy_e_retr)
    np.save(PATH_PRE + PATHTYPE + letters[0] + "/reorder_xy_e_rep.npy", xy_e_rep)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_s.npy", d_lat_s)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_s.npy", d_cat_s)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_lat_e.npy", d_lat_e)
    np.save(PATH_PRE+PATHTYPE+letters[0]+"/reorder_d_cat_e.npy", d_cat_e)

dmean_s = np.mean(d_s,axis=0)
dmean_e_retr = np.mean(d_e_retr,axis=0)
dmean_e_rep = np.mean(d_e_rep,axis=0)
dstd_s = np.std(d_s,axis=0)
dstd_e_retr = np.std(d_e_retr,axis=0)
dstd_e_rep = np.std(d_e_rep,axis=0)

rcatmean_s = np.mean(rcat_s,axis=0)
rcatmean_e_retr = np.mean(rcat_e_retr,axis=0)
rcatmean_e_rep = np.mean(rcat_e_rep,axis=0)
rcatstd_s = np.std(rcat_s,axis=0)
rcatstd_e_retr = np.std(rcat_e_retr,axis=0)
rcatstd_e_rep = np.std(rcat_e_rep,axis=0)

xymean_s = np.mean(xy_s,axis=0)
xymean_e_retr = np.mean(xy_e_retr,axis=0)
xymean_e_rep = np.mean(xy_e_rep,axis=0)
xystd_s = np.std(xy_s,axis=0)
xystd_e_retr = np.std(xy_e_retr,axis=0)
xystd_e_rep = np.std(xy_e_rep,axis=0)

d_latmean_s = np.mean(d_lat_s,axis=0)
d_latmean_e = np.mean(d_lat_e,axis=0)
d_latstd_s = np.std(d_lat_s,axis=0)
d_latstd_e = np.std(d_lat_e,axis=0)

d_catmean_s = np.mean(d_cat_s,axis=0)
d_catmean_e = np.mean(d_cat_e,axis=0)
d_catstd_s = np.std(d_cat_s,axis=0)
d_catstd_e = np.std(d_cat_e,axis=0)

# TODO: plotting

f, ax = plt.subplots(3,1, sharex=True)
ax[0].plot(xlbls, np.transpose(dmean_s)[0], label="simple0", c='g', ls='-')
ax[0].plot(xlbls, np.transpose(dmean_e_retr)[0], label="retr0", c='r', ls='-')
ax[0].plot(xlbls, np.transpose(dmean_e_rep)[0], label="rep0", c='m', ls='-')
ax[0].plot(xlbls, np.transpose(dmean_s)[REPS], label="simple39", c='g', ls=':')
ax[0].plot(xlbls, np.transpose(dmean_e_retr)[REPS], label="retr39", c='r', ls=':')
ax[0].plot(xlbls, np.transpose(dmean_e_rep)[REPS], label="rep39", c='m', ls=':')
# ax[0].fill_between(xlbls, dmean_s-dstd_s, dmean_s+dstd_s, facecolor='g', alpha=0.1)
# ax[0].fill_between(xlbls, dmean_e-dstd_e, dmean_e+dstd_e, facecolor='r', alpha=0.1)
ax[0].set_xscale('log')
ax[0].set_xticks([2, 10, 100, 600])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax[0].set_xlabel("Length of training sequences")
ax[0].set_ylabel("Delta value of SFA output")
# ax[0].legend()
# ax[0].plot(xlbls[0], dmean_s[0], c="k", mfc="none", marker="o", markersize=12)
# ax[0].plot(xlbls[0], dmean_e[0], c="k", mfc="none", marker="s", markersize=12)
# ax[0].plot(xlbls[0], dmean_s[0], c="b", mfc="none", marker="+", markersize=12)
# ax[0].plot(xlbls[25], dmean_s[25], c="b", mfc="none", marker="x", markersize=12)

# lc1, = ax[1].plot(xlbls, catmean_s, c='g', ls='--')
# lc2, = ax[1].plot(xlbls, catmean_e, c='r', ls='--')
# ax[1].fill_between(xlbls, catmean_s-catstd_s, catmean_s+catstd_s, facecolor='g', alpha=0.1)
# ax[1].fill_between(xlbls, catmean_e-catstd_e, catmean_e+catstd_e, facecolor='r', alpha=0.1)
lc1r, = ax[2].plot(xlbls, np.transpose(rcatmean_s)[0], c='g', ls='-')
lc2r, = ax[2].plot(xlbls, np.transpose(rcatmean_e_retr)[0], c='r', ls='-')
lc3r, = ax[2].plot(xlbls, np.transpose(rcatmean_e_rep)[0], c='m', ls='-')
lc4r, = ax[2].plot(xlbls, np.transpose(rcatmean_s)[REPS], c='g', ls=':')
lc5r, = ax[2].plot(xlbls, np.transpose(rcatmean_e_retr)[REPS], c='r', ls=':')
lc6r, = ax[2].plot(xlbls, np.transpose(rcatmean_e_rep)[REPS], c='m', ls=':')
# ax[1].fill_between(xlbls, rcatmean_s-rcatstd_s, rcatmean_s+rcatstd_s, facecolor='g', alpha=0.1)
# ax[1].fill_between(xlbls, rcatmean_e-rcatstd_e, rcatmean_e+rcatstd_e, facecolor='r', alpha=0.1)
l1, = ax[1].plot(xlbls, np.transpose(xymean_s)[0], label="simple0", c='g', ls='-')
l2, = ax[1].plot(xlbls, np.transpose(xymean_e_retr)[0], label="retr0", c='r', ls='-')
l3, = ax[1].plot(xlbls, np.transpose(xymean_e_rep)[0], label="rep0", c='m', ls='-')
l4, = ax[1].plot(xlbls, np.transpose(xymean_s)[REPS], label="simple39", c='g', ls=':')
l5, = ax[1].plot(xlbls, np.transpose(xymean_e_retr)[REPS], label="retr39", c='r', ls=':')
l6, = ax[1].plot(xlbls, np.transpose(xymean_e_rep)[REPS], label="rep39", c='m', ls=':')
# ax[1].fill_between(xlbls, xymean_s-xystd_s, xymean_s+xystd_s, facecolor='g', alpha=0.1)
# ax[1].fill_between(xlbls, xymean_e-xystd_e, xymean_e+xystd_e, facecolor='r', alpha=0.1)
ax[1].set_xscale('log')
ax[1].set_xticks([2, 10, 100, 600])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_ylabel("x/y prediction")
ax[2].set_ylabel("obeject identity prediction")
# ax[1].plot(xlbls[0], catmean_s[0], c="k", mfc="none", marker="o", markersize=12)
# ax[1].plot(xlbls[0], catmean_e[0], c="k", mfc="none", marker="s", markersize=12)
# ax[1].plot(xlbls[0], xymean_s[0], c="k", mfc="none", marker="o", markersize=12)
# ax[1].plot(xlbls[0], xymean_e[0], c="k", mfc="none", marker="s", markersize=12)
# ax[1].plot(xlbls[0], rcatmean_s[0], c="k", mfc="none", marker="o", markersize=12)
# ax[1].plot(xlbls[0], rcatmean_e[0], c="k", mfc="none", marker="s", markersize=12)
# ax[1].plot(xlbls[0], catmean_s[0], c="b", mfc="none", marker="+", markersize=12)
# ax[1].plot(xlbls[25], catmean_s[25], c="b", mfc="none", marker="x", markersize=12)
# ax[1].plot(xlbls[0], xymean_s[0], c="b", mfc="none", marker="+", markersize=12)
# ax[1].plot(xlbls[25], xymean_s[25], c="b", mfc="none", marker="x", markersize=12)
# ax[1].plot(xlbls[0], rcatmean_s[0], c="b", mfc="none", marker="+", markersize=12)
# ax[1].plot(xlbls[25], rcatmean_s[25], c="b", mfc="none", marker="x", markersize=12)
# ax[1].legend((lc1, lc1r,l1), ("object category", "cat regression r²", "x/r regression r²"), loc=4)
# ax[1].legend((l1, lc1r), ("x/y", "category"), loc=4)
# f.subplots_adjust(bottom=0.15)
#
# ldl, = ax[2].plot(xlbls, d_latmean_s, c='g', ls='-')
# ldc, = ax[2].plot(xlbls, d_catmean_s, c='g', ls='--')
# # ax[2].fill_between(xlbls, d_latmean_s-d_latstd_s, d_latmean_s+d_latstd_s, facecolor='g', alpha=0.1)
# # ax[2].fill_between(xlbls, d_catmean_s-d_catstd_s, d_catmean_s-d_catstd_s, facecolor='g', alpha=0.1)
# # ax[2].plot(xlbls[0], d_catmean_s[0], c="k", mfc="none", marker="o", markersize=12)
# # ax[2].plot(xlbls[0], d_catmean_e[0], c="k", mfc="none", marker="s", markersize=12)
# # ax[2].plot(xlbls[0], d_latmean_s[0], c="k", mfc="none", marker="o", markersize=12)
# # ax[2].plot(xlbls[0], d_latmean_e[0], c="k", mfc="none", marker="s", markersize=12)
# ax[2].plot(xlbls, d_latmean_e, c='r', ls='-')
# ax[2].plot(xlbls, d_catmean_e, c='r', ls='--')
# # ax[2].fill_between(xlbls, d_latmean_e-d_latstd_e, d_latmean_e-d_latstd_e, facecolor='r', alpha=0.1)
# # ax[2].fill_between(xlbls, d_catmean_e-d_catstd_e, d_catmean_e-d_catstd_e, facecolor='r', alpha=0.1)
# # ax[2].plot(xlbls[0], d_catmean_s[0], c="b", mfc="none", marker="+", markersize=12)
# # ax[2].plot(xlbls[25], d_catmean_s[25], c="b", mfc="none", marker="x", markersize=12)
# # ax[2].plot(xlbls[0], d_latmean_s[0], c="b", mfc="none", marker="+", markersize=12)
# # ax[2].plot(xlbls[25], d_latmean_s[25], c="b", mfc="none", marker="x", markersize=12)
# ax[2].legend((ldl, ldc), ("x/y", "category"))
# ax[2].set_ylabel("Delta value of latents")
# ax[2].set_xlabel("Length of training sequences")

plt.figlegend((l1,l2,l3,l4,l5,l6), ("simple0", "retr0", "rep0", "simple{}".format(REPS), "retr{}".format(REPS), "rep{}".format(REPS)), 1)
# plt.figlegend((lc1,l1), ("object category", "x/r regression r²"), 2)
plt.show()
