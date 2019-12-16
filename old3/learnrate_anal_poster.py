from core import semantic, system_params, input_params, tools, streamlined, result

import numpy as np
import sys
import pickle
from matplotlib import pyplot as plt

PLOT_XY = False
PLOT_PHI = False

TEST_DATA = 1  # 0: Training only, 1: New only, 2: Both
LEARNRATE2 = True

PATH = "/local/results/learnrate2/"

# colors = [['b', 'c'], ['r', 'm'], ['g', 'y'], ['k', '0.5']]
colors = ['b', 'r', 'g', 'k']
linest = ['--', '-']

DCNT = 4
SINGLED = False

NRUNS = 40
eps_list = [0.0001, 0.001, 0.01, 0.1]
NSEQS = TEST_DATA if TEST_DATA > 0 else 1
NEPS = len(eps_list)

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1['number_of_snippets'] = 50
if not LEARNRATE2:
    PARAMETERS.st1['input_noise'] = 0.2
    PARAMETERS.st1['frame_shape'] = (12,12)
    PARAMETERS.st1['object_code'] = input_params.make_object_code(['-'], sizes=22)
    PARAMETERS.st1['sequence'] = [0]

sample_parms = dict(PARAMETERS.st1)
sample_parms['number_of_snippets'] = 1
sample_parms['movement_type'] = 'sample'
sample_parms['movement_params'] = dict()

sensory_system = pickle.load(open(PATH + "sensory.p", 'rb'))
ran = np.arange(PARAMETERS.st1['number_of_snippets'])

training_sequence = training_latent = training_categories = new_sequence1 = new_sequence2 = new_latent = new_categories = 0
b1_d_n = b1_d_t = b2_d_n = b2_d_t = b1_xycorr_n = b1_xycorr_t = b2_xycorr_n = b2_xycorr_t = b1_phicorr_n = b1_phicorr_t = b2_phicorr_n = b2_phicorr_t = 0
xy_d_n = xy_d_t = phi_d_n = phi_d_t = 0

if TEST_DATA == 0 or TEST_DATA == 2:
    training_sequence, training_categories, training_latent = sensory_system.recall(numbers = ran, fetch_indices=False, **PARAMETERS.st1)
if TEST_DATA > 0:
    new_sequence1, new_categories, new_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)
    input_delta = np.mean(tools.delta_diff(new_sequence1))
    new_sequence2 = np.array(new_sequence1)
    if LEARNRATE2:
        sfa1 = semantic.load_SFA("/local/results/learnrate2/sfa1.p")
        sfa2 = semantic.load_SFA("/local/results/learnrate2/sfa2.p")
        new_sequence1 = sfa1(new_sequence1)
        new_sequence1 = streamlined.normalizer(new_sequence1, PARAMETERS.normalization)(new_sequence1)
        intermediate_delta1 = np.mean(np.sort(tools.delta_diff(new_sequence1))[:DCNT])
        new_sequence2 = sfa2(new_sequence2)
        new_sequence2 = streamlined.normalizer(new_sequence2, PARAMETERS.normalization)(new_sequence2)
        intermediate_delta2 = np.mean(np.sort(tools.delta_diff(new_sequence2))[:DCNT])

b1_sfa = semantic.load_SFA(PATH + "b1.sfa")
b2_sfa = semantic.load_SFA(PATH + "b2.sfa")

if TEST_DATA == 0 or TEST_DATA == 2:
    b1_y_t = semantic.exec_SFA(b1_sfa, training_sequence)
    b1_w_t = streamlined.normalizer(b1_y_t, PARAMETERS.normalization)(b1_y_t)
    b2_y_t = semantic.exec_SFA(b2_sfa, training_sequence)
    b2_w_t = streamlined.normalizer(b2_y_t, PARAMETERS.normalization)(b2_y_t)
    b1_ds_t = np.sort(tools.delta_diff(b1_w_t))[:DCNT]
    b2_ds_t = np.sort(tools.delta_diff(b2_w_t))[:DCNT]
    if SINGLED:
        b1_d_t = []
        b2_d_t = []
        for di in range(DCNT):
            b1_d_t.append(b1_ds_t[di])
            b2_d_t.append(b2_ds_t[di])
    else:
        b1_d_t = np.mean(b1_ds_t)
        b2_d_t = np.mean(b2_ds_t)

if TEST_DATA > 0:
    b1_y_n = semantic.exec_SFA(b1_sfa, new_sequence1)
    b1_w_n = streamlined.normalizer(b1_y_n, PARAMETERS.normalization)(b1_y_n)
    b2_y_n = semantic.exec_SFA(b2_sfa, new_sequence2)
    b2_w_n = streamlined.normalizer(b2_y_n, PARAMETERS.normalization)(b2_y_n)
    b1_ds_n = np.sort(tools.delta_diff(b1_w_n))[:DCNT]
    b2_ds_n = np.sort(tools.delta_diff(b2_w_n))[:DCNT]
    if SINGLED:
        b1_d_n = []
        b2_d_n = []
        for di in range(DCNT):
            b1_d_n.append(b1_ds_n[di])
            b2_d_n.append(b2_ds_n[di])
    else:
        b1_d_n = np.mean(b1_ds_n)
        b2_d_n = np.mean(b2_ds_n)

if TEST_DATA == 0 or TEST_DATA == 2:
    x_d_t = tools.delta_diff(training_latent)[0]
    y_d_t = tools.delta_diff(training_latent)[1]
    xy_d_t = np.mean([x_d_t, y_d_t])
    cos_d_t = tools.delta_diff(training_latent)[2]
    sin_d_t = tools.delta_diff(training_latent)[3]
    phi_d_t = np.mean([cos_d_t, sin_d_t])
if TEST_DATA > 0:
    x_d_n = tools.delta_diff(new_latent)[0]
    y_d_n = tools.delta_diff(new_latent)[1]
    xy_d_n = np.mean([x_d_n, y_d_n])
    cos_d_n = tools.delta_diff(new_latent)[2]
    sin_d_n = tools.delta_diff(new_latent)[3]
    phi_d_n = np.mean([cos_d_n, sin_d_n])

if TEST_DATA == 0 or TEST_DATA == 2:
    b1_corr_t = np.abs(tools.feature_latent_correlation(b1_w_t, training_latent, training_categories))
    b1_xycorr_t = np.mean([np.max(b1_corr_t[0,:]),np.max(b1_corr_t[1,:])])
    b1_phicorr_t = np.mean([np.max(b1_corr_t[2,:]),np.max(b1_corr_t[3,:])])
    b2_corr_t = np.abs(tools.feature_latent_correlation(b2_w_t, training_latent, training_categories))
    b2_xycorr_t = np.mean([np.max(b2_corr_t[0,:]),np.max(b2_corr_t[1,:])])
    b2_phicorr_t = np.mean([np.max(b2_corr_t[2,:]),np.max(b2_corr_t[3,:])])
if TEST_DATA > 0:
    b1_corr_n = np.abs(tools.feature_latent_correlation(b1_w_n, new_latent, new_categories))
    b1_xycorr_n = np.mean([np.max(b1_corr_n[0,:]),np.max(b1_corr_n[1,:])])
    b1_phicorr_n = np.mean([np.max(b1_corr_n[2,:]),np.max(b1_corr_n[3,:])])
    b2_corr_n = np.abs(tools.feature_latent_correlation(b2_w_n, new_latent, new_categories))
    b2_xycorr_n = np.mean([np.max(b2_corr_n[0,:]),np.max(b2_corr_n[1,:])])
    b2_phicorr_n = np.mean([np.max(b2_corr_n[2,:]),np.max(b2_corr_n[3,:])])

x = list(range(NRUNS))

fd, ad = plt.subplots(NSEQS, NEPS, squeeze=False, sharex=True, sharey=True)
fd.text(0.5, 0.04, 'number of training repetitions', ha='center', va='center')
if SINGLED:
    fd.text(0.06, 0.5, 'delta-value (individual features)'.format(DCNT), ha='center', va='center', rotation='vertical')
else:
    fd.text(0.06, 0.5, 'delta-value (average of {} slowest features)'.format(DCNT), ha='center', va='center', rotation='vertical')
if PLOT_XY:
    fxy, axy = plt.subplots(NSEQS, NEPS, squeeze=False, sharex=True, sharey=True)
    fxy.text(0.5, 0.04, 'number of training repetitions', ha='center', va='center')
    fxy.text(0.06, 0.5, 'correlation of features with x-y-coordinate', ha='center', va='center', rotation='vertical')
if PLOT_PHI:
    fphi, aphi = plt.subplots(NSEQS, NEPS, squeeze=False, sharex=True, sharey=True)
    fphi.text(0.5, 0.04, 'number of training repetitions', ha='center', va='center')
    fphi.text(0.06, 0.5, 'correlation of features with rotation angle', ha='center', va='center', rotation='vertical')

corlst1 = []
corlst2 = []
for iseq in range(NSEQS):
    for ei, eps in enumerate(eps_list):
        inc1_d = []
        inc2_d = []
        if PLOT_XY:
            inc1_xycorr = []
            inc2_xycorr = []
        if PLOT_PHI:
            inc1_phicorr = []
            inc2_phicorr = []
        for i in range(NRUNS):
            inc1_sfa = semantic.load_SFA(PATH + "inc1_eps{}_{}.sfa".format(ei, i))
            inc2_sfa = semantic.load_SFA(PATH + "inc2_eps{}_{}.sfa".format(ei, i))

            inc1_y = semantic.exec_SFA(inc1_sfa, [training_sequence, new_sequence1][iseq+TEST_DATA%2])
            inc1_w = streamlined.normalizer(inc1_y, PARAMETERS.normalization)(inc1_y)
            inc1_ds = np.sort(tools.delta_diff(inc1_w))[:DCNT]
            inc2_y = semantic.exec_SFA(inc2_sfa, [training_sequence, new_sequence2][iseq+TEST_DATA%2])
            inc2_w = streamlined.normalizer(inc2_y, PARAMETERS.normalization)(inc2_y)
            inc2_ds = np.sort(tools.delta_diff(inc2_w))[:DCNT]
            if SINGLED:
                for di in range(DCNT):
                    if i == 0:
                        inc1_d.append([])
                        inc2_d.append([])
                    inc1_d[di].append(inc1_ds[di])
                    inc2_d[di].append(inc2_ds[di])
            else:
                inc1_d.append(np.mean(inc1_ds))
                inc2_d.append(np.mean(inc2_ds))
            if PLOT_XY or PLOT_PHI:
                inc1_corr_raw = tools.feature_latent_correlation(inc1_w, [training_latent, new_latent][iseq + TEST_DATA % 2], [training_categories, new_categories][iseq + TEST_DATA % 2])
                inc1_corr = np.abs(inc1_corr_raw)
                inc2_corr_raw = tools.feature_latent_correlation(inc2_w, [training_latent, new_latent][iseq + TEST_DATA % 2], [training_categories, new_categories][iseq + TEST_DATA % 2])
                inc2_corr = np.abs(inc2_corr_raw)
                if ei == 1:
                    corlst1.append(inc1_corr_raw)
                    corlst2.append(inc2_corr_raw)
            if PLOT_XY:
                inc1_xycorr.append(np.mean([np.max(inc1_corr[0, :]), np.max(inc1_corr[1, :])]))
                inc2_xycorr.append(np.mean([np.max(inc2_corr[0, :]), np.max(inc2_corr[1, :])]))
            if PLOT_PHI:
                inc1_phicorr.append(np.mean([np.max(inc1_corr[2, :]), np.max(inc1_corr[3, :])]))
                inc2_phicorr.append(np.mean([np.max(inc2_corr[2, :]), np.max(inc2_corr[3, :])]))
        axd = ad[iseq][ei]
        fd.suptitle("avg d-values")
        axd.set_title(["train_seq, ", "new_seq, ", ""][iseq+2*int(TEST_DATA < 2)] + "eps = {}".format(eps))
        if SINGLED:
            for di in range(DCNT):
                axd.plot(x, [[b1_d_t[di], b1_d_n[di]][iseq+TEST_DATA%2]] * NRUNS, label="b1", c=colors[0], ls=linest[0])
                axd.plot(x, inc1_d[di], label="inc1", c=colors[1], ls=linest[0])
                axd.plot(x, [[b2_d_t[di], b2_d_n[di]][iseq+TEST_DATA%2]] * NRUNS, label="b2", c=colors[0], ls=linest[1])
                axd.plot(x, inc2_d[di], label="inc2", c=colors[1], ls=linest[1])
        else:
            axd.plot(x, [[b1_d_t, b1_d_n][iseq+TEST_DATA%2]] * NRUNS, label="b1", c=colors[0], ls=linest[0])
            axd.plot(x, inc1_d, label="inc1", c=colors[1], ls=linest[0])
            axd.plot(x, [[b2_d_t, b2_d_n][iseq+TEST_DATA%2]] * NRUNS, label="b2", c=colors[0], ls=linest[1])
            axd.plot(x, inc2_d, label="inc2", c=colors[1], ls=linest[1])
        if LEARNRATE2:
            axd.plot(x, [intermediate_delta1]*NRUNS, label="pre1", c=colors[2], ls=linest[0])
            axd.plot(x, [intermediate_delta2]*NRUNS, label="pre2", c=colors[2], ls=linest[1])
        axd.plot(x, [[xy_d_t, xy_d_n][iseq+TEST_DATA%2]]*NRUNS, label="xy", c=colors[-1], ls=linest[0])
        axd.plot(x, [[phi_d_t, phi_d_n][iseq+TEST_DATA%2]] * NRUNS, label="phi", c=colors[-1], ls=linest[1])
        axd.legend()
        if ei == 1:
            inc1_ddd = inc1_d

        if PLOT_XY:
            axxy = axy[iseq][ei]
            fxy.suptitle("mean of highest feature correlation with x and with y")
            axxy.set_title(["train_seq, ", "new_seq, ", ""][iseq+2*int(TEST_DATA < 2)] + "eps = {}".format(eps))
            axxy.plot(x, [[b1_xycorr_t, b1_xycorr_n][iseq+TEST_DATA%2]] * NRUNS, label="b1", c=colors[0], ls=linest[0])
            axxy.plot(x, inc1_xycorr, label="inc1", c=colors[1], ls=linest[0])
            axxy.plot(x, [[b2_xycorr_t, b2_xycorr_n][iseq+TEST_DATA%2]] * NRUNS, label="b2", c=colors[0], ls=linest[1])
            axxy.plot(x, inc2_xycorr, label="inc2", c=colors[1], ls=linest[1])
            axxy.legend()

        if PLOT_PHI:
            axphi = aphi[iseq][ei]
            fphi.suptitle("mean of highest feature correlation with sin(phi) and with cos(phi)")
            axphi.set_title(["train_seq, ", "new_seq, ", ""][iseq+2*int(TEST_DATA < 2)] + "eps = {}".format(eps))
            axphi.plot(x, [[b1_phicorr_t, b1_phicorr_n][iseq+TEST_DATA%2]] * NRUNS, label="b1", c=colors[0], ls=linest[0])
            axphi.plot(x, inc1_phicorr, label="inc1", c=colors[1], ls=linest[0])
            axphi.plot(x, [[b2_phicorr_t, b2_phicorr_n][iseq+TEST_DATA%2]] * NRUNS, label="b2", c=colors[0], ls=linest[1])
            axphi.plot(x, inc2_phicorr, label="inc2", c=colors[1], ls=linest[1])
            axphi.legend()

fp, axp = plt.subplots()
axp.plot(x, [b1_d_n] * NRUNS, label="batch SFA2", c=colors[0], ls=linest[0])
axp.plot(x, inc1_ddd, label="incremental SFA2", c=colors[1], ls=linest[0])
axp.plot(x, [intermediate_delta1]*NRUNS, label="SFA1", c=colors[2], ls=linest[0])
axp.set_xlabel("Number of training repetitions")
axp.set_ylabel("Delta value of SFA output")
axp.legend()
plt.gcf().subplots_adjust(bottom=0.25)

plt.show()
