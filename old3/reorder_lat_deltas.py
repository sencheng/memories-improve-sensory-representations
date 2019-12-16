import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from core import semantic, streamlined
import scipy.stats

xlbls = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
dirname = "/local/results/reorder4_new/"
testing_data = np.load(dirname + "testing0.npz")
seq2, cat2, lat2, _ = testing_data['testing_sequenceX'], testing_data['testing_categories'], testing_data['testing_latent'], testing_data['testing_ranges']

d_s, d_e, cat_s, cat_e, rcat_s, rcat_e, xy_s, xy_e = [], [], [], [], [], [], [], []
d_lat_s, d_cat_s, d_lat_e, d_cat_e = [], [], [], []

for ri in range(31):
    print(ri)
    fname = "res{}.p".format(ri)
    with open(dirname + fname, 'rb') as f:
        res = pickle.load(f)
    d_s.append(np.mean(np.sort(res.d_values['testingZ_S'])[:3]))
    d_e.append(np.mean(np.sort(res.d_values['testingZ_E'])[:3]))

    d_lat_s.append(np.mean(res.d_values['forming_lat'][:2]))
    d_cat_s.append(np.mean(res.d_values['forming_cat']))
    d_lat_e.append(np.mean(res.d_values['retrieved_lat'][:2]))
    d_cat_e.append(np.mean(res.d_values['retrieved_cat']))

    cat_s.append(np.max(np.abs(res.testing_corrZ_simple[4, :])))
    cat_e.append(np.max(np.abs(res.testing_corrZ_episodic[4, :])))

    sfa1 = semantic.load_SFA(dirname + res.data_description + "train0.sfa")
    yy2 = semantic.exec_SFA(sfa1, seq2)
    yy2_w = streamlined.normalizer(yy2, res.params.normalization)(yy2)
    zz2S = semantic.exec_SFA(res.sfa2S, yy2_w)
    zz2S_w = streamlined.normalizer(zz2S, res.params.normalization)(zz2S)
    zz2E = semantic.exec_SFA(res.sfa2E, yy2_w)
    zz2E_w = streamlined.normalizer(zz2E, res.params.normalization)(zz2E)

    predictionS = res.learnerS.predict(zz2S_w)
    predictionE = res.learnerE.predict(zz2E_w)
    _, _, r_valueX_S, _, _ = scipy.stats.linregress(lat2[:, 0], predictionS[:, 0])
    _, _, r_valueY_S, _, _ = scipy.stats.linregress(lat2[:, 1], predictionS[:, 1])
    _, _, r_valueCAT_S, _, _ = scipy.stats.linregress(cat2, predictionS[:, 4])
    # _, _, r_valueCosphi_S, _, _ = scipy.stats.linregress(lat2[:, 2], predictionS[:, 2])
    # _, _, r_valueSinphi_S, _, _ = scipy.stats.linregress(lat2[:, 3], predictionS[:, 3])
    _, _, r_valueX_E, _, _ = scipy.stats.linregress(lat2[:, 0], predictionE[:, 0])
    _, _, r_valueY_E, _, _ = scipy.stats.linregress(lat2[:, 1], predictionE[:, 1])
    _, _, r_valueCAT_E, _, _ = scipy.stats.linregress(cat2, predictionE[:, 4])
    # _, _, r_valueCosphi_E, _, _ = scipy.stats.linregress(lat2[:, 2], predictionE[:, 2])
    # _, _, r_valueSinphi_E, _, _ = scipy.stats.linregress(lat2[:, 3], predictionE[:, 3])

    xy_s.append(np.mean((r_valueX_S, r_valueY_S)))
    rcat_s.append(r_valueCAT_S)
    # phiZ_S.append(np.mean((r_valueCosphi_S, r_valueSinphi_S)))
    xy_e.append(np.mean((r_valueX_E, r_valueY_E)))
    rcat_e.append(r_valueCAT_E)

dmean_s = d_s
dmean_e = d_e
catmean_s = cat_s
catmean_e = cat_e
rcatmean_s = rcat_s
rcatmean_e = rcat_e
xymean_s = xy_s
xymean_e = xy_e

f, ax = plt.subplots(3,1, sharex=True)
ax[0].plot(xlbls, dmean_s, label="simple", c='g', ls='-')
ax[0].plot(xlbls, dmean_e, label="episodic", c='r', ls='-')
ax[0].set_xscale('log')
ax[0].set_xticks([2, 10, 100, 600])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax[0].set_xlabel("Length of training sequences")
ax[0].set_ylabel("Delta value of SFA output")
# ax[0].legend()
ax[0].plot(xlbls[0], dmean_s[0], c="k", mfc="none", marker="o", markersize=12)
ax[0].plot(xlbls[0], dmean_e[0], c="k", mfc="none", marker="s", markersize=12)
ax[0].plot(xlbls[0], dmean_s[0], c="b", mfc="none", marker="+", markersize=12)
ax[0].plot(xlbls[25], dmean_s[25], c="b", mfc="none", marker="x", markersize=12)

lc1, = ax[1].plot(xlbls, catmean_s, c='g', ls='--')
lc2, = ax[1].plot(xlbls, catmean_e, c='r', ls='--')
lc1r, = ax[1].plot(xlbls, rcatmean_s, c='g', ls='-.')
lc2r, = ax[1].plot(xlbls, rcatmean_e, c='r', ls='-.')
l1, = ax[1].plot(xlbls, xymean_s, label="simple", c='g', ls='-')
l2, = ax[1].plot(xlbls, xymean_e, label="episodic", c='r', ls='-')
ax[1].set_xscale('log')
ax[1].set_xticks([2, 10, 100, 600])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_ylabel("correlation value")
ax[1].plot(xlbls[0], catmean_s[0], c="k", mfc="none", marker="o", markersize=12)
ax[1].plot(xlbls[0], catmean_e[0], c="k", mfc="none", marker="s", markersize=12)
ax[1].plot(xlbls[0], xymean_s[0], c="k", mfc="none", marker="o", markersize=12)
ax[1].plot(xlbls[0], xymean_e[0], c="k", mfc="none", marker="s", markersize=12)
ax[1].plot(xlbls[0], catmean_s[0], c="b", mfc="none", marker="+", markersize=12)
ax[1].plot(xlbls[25], catmean_s[25], c="b", mfc="none", marker="x", markersize=12)
ax[1].plot(xlbls[0], xymean_s[0], c="b", mfc="none", marker="+", markersize=12)
ax[1].plot(xlbls[25], xymean_s[25], c="b", mfc="none", marker="x", markersize=12)
ax[1].legend((lc1, lc1r,l1), ("object category", "cat regression r²", "x/r regression r²"), loc=4)

ldl, = ax[2].plot(xlbls, d_lat_s, c='g', ls='-')
ldc, = ax[2].plot(xlbls, d_cat_s, c='g', ls='--')
ax[2].plot(xlbls, d_lat_e, c='r', ls='-')
ax[2].plot(xlbls, d_cat_e, c='r', ls='--')
ax[2].legend((ldl, ldc), ("x/y", "category"))
ax[2].set_ylabel("Delta value of latents")
ax[2].set_xlabel("Length of training sequences")

# f.subplots_adjust(bottom=0.15)
plt.figlegend((l1,l2), ("simple", "episodic"), 1)
# plt.figlegend((lc1,l1), ("object category", "x/r regression r²"), 2)
plt.show()