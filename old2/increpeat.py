from core import semantic, sensory, system_params, input_params, tools, streamlined

import numpy as np
import pickle
from matplotlib import pyplot as plt
import sys

if len(sys.argv) > 1:
    PATH = "/local/results/increp_{}/".format(sys.argv[1])
else:
    PATH = "/local/results/increp_old/increp_40/"
if len(sys.argv) > 2:
    XCNT = int(sys.argv[2])
else:
    XCNT = 31
XNUM = 31

# colors = [['b', 'c'], ['r', 'm'], ['g', 'y'], ['k', '0.5']]
colors = ['b', 'r', 'g', 'k']
linest = ['--', '-']
DCNT = 4
REPS = [0,3,9,19]    # plot only the reps in the list. plot all when empty list

REPNUM = 20 if REPS == [] else len(REPS)
replist = range(REPNUM) if REPS == [] else list(REPS)

dZ_S1 = []
dZ_E1 = []
xyZ_S1 = []
phiZ_S1 = []
xyZ_E1 = []
phiZ_E1 = []
x = []

fname = "{}res{}.p".format(PATH, 0)
res = pickle.load(open(fname, 'rb'))
testzip = np.load("{}testing0.npz".format(PATH))
testing_sequenceX, testing_categories, testing_latent, testing_ranges = testzip['testing_sequenceX'], testzip['testing_categories'], testzip['testing_latent'], testzip['testing_ranges']
SFA1 = semantic.load_SFA('{}sfadef2train0.sfa'.format(PATH))
testing_sequenceY = semantic.exec_SFA(SFA1, testing_sequenceX)
testing_sequenceY = streamlined.normalizer(testing_sequenceY, res.params.normalization)(testing_sequenceY)

testing_d = tools.delta_diff(res.testing_latent)
xy_d = np.mean([testing_d[0], testing_d[1]])
phi_d = np.mean([testing_d[2], testing_d[3]])

for i in range(XCNT):
    fname = "{}res{}.p".format(PATH, i)
    res = pickle.load(open(fname, 'rb'))
    vS = res.d_values['testingZ_S']
    dZ_S1.append(np.mean(np.sort(vS)[:DCNT]) if isinstance(vS, np.ndarray) else vS)
    vE = res.d_values['testingZ_E']
    dZ_E1.append(np.mean(np.sort(vE)[:DCNT]) if isinstance(vE, np.ndarray) else vE)
    x.append(res.params.st2["snippet_length"])

    cS = tools.feature_latent_correlation(res.testingZ_S, testing_latent)
    xyZ_S1.append(np.mean([np.max(cS[0, :]), np.max(cS[1, :])]))
    phiZ_S1.append(np.mean([np.max(cS[2, :]), np.max(cS[3, :])]))
    cE = tools.feature_latent_correlation(res.testingZ_E, res.testing_latent)
    xyZ_E1.append(np.mean([np.max(cE[0, :]), np.max(cE[1, :])]))
    phiZ_E1.append(np.mean([np.max(cE[2, :]), np.max(cE[3, :])]))

dz_S = []
dz_E = []
xyz_S = []
phiz_S = []
xyz_E = []
phiz_E = []
for j, rep in enumerate(replist):
    dz_S.append([])
    dz_E.append([])
    xyz_S.append([])
    phiz_S.append([])
    xyz_E.append([])
    phiz_E.append([])
    for i in range(XNUM, XNUM+XCNT):
        SFA2S = semantic.load_SFA('{}sfa2S_res{}_repeat{}.sfa'.format(PATH, i, rep))
        seqZ_S = semantic.exec_SFA(SFA2S, testing_sequenceY)
        seqZ_S_w = streamlined.normalizer(seqZ_S, res.params.normalization)(seqZ_S)
        S_ds = np.sort(tools.delta_diff(seqZ_S_w))[:DCNT]
        dz_S[j].append(np.mean(S_ds))
        SFA2E = semantic.load_SFA('{}sfa2E_res{}_repeat{}.sfa'.format(PATH, i, rep))
        seqZ_E = semantic.exec_SFA(SFA2E, testing_sequenceY)
        seqZ_E_w = streamlined.normalizer(seqZ_E, res.params.normalization)(seqZ_E)
        E_ds = np.sort(tools.delta_diff(seqZ_E_w))[:DCNT]
        dz_E[j].append(np.mean(E_ds))

        cS = tools.feature_latent_correlation(seqZ_S_w, testing_latent)
        xyz_S[j].append(np.mean([np.max(cS[0, :]), np.max(cS[1, :])]))
        phiz_S[j].append(np.mean([np.max(cS[2, :]), np.max(cS[3, :])]))
        cE = tools.feature_latent_correlation(seqZ_E_w, res.testing_latent)
        xyz_E[j].append(np.mean([np.max(cE[0, :]), np.max(cE[1, :])]))
        phiz_E[j].append(np.mean([np.max(cE[2, :]), np.max(cE[3, :])]))

cols = int(np.sqrt(REPNUM))
rows = int(REPNUM / cols) + int(bool(REPNUM % cols))
f, ax = plt.subplots(cols, rows, squeeze=False, sharex=True, sharey=True)
f.text(0.5, 0.04, 'length of training snippets', ha='center', va='center')
f.text(0.06, 0.5, 'delta-value (average of {} slowest features)'.format(DCNT), ha='center', va='center', rotation='vertical')
fxy, axxy = plt.subplots(cols, rows, squeeze=False, sharex=True, sharey=True)
fxy.text(0.5, 0.04, 'length of training snippets', ha='center', va='center')
fxy.text(0.06, 0.5, 'correlation of features with x-y-coordinate', ha='center', va='center', rotation='vertical')
fphi, axphi = plt.subplots(cols, rows, squeeze=False, sharex=True, sharey=True)
fphi.text(0.5, 0.04, 'length of training snippets', ha='center', va='center')
fphi.text(0.06, 0.5, 'correlation of features with rotation angle', ha='center', va='center', rotation='vertical')
for i, rep in enumerate(replist):
    ci = i // rows
    ri = i % rows
    axc = ax[ci, ri]
    f.suptitle("d-values")
    axc.set_title("REP {}".format(rep))
    axc.set_xscale('log')
    axc.plot(x, dZ_S1, label="bSFA_S", c=colors[0], ls=linest[0])
    axc.plot(x, dZ_E1, label="bSFA_E", c=colors[0], ls=linest[1])
    axc.plot(x, dz_S[i], label="incSFA_S", c=colors[1], ls=linest[0])
    axc.plot(x, dz_E[i], label="incSFA_E", c=colors[1], ls=linest[1])
    axc.plot(x, [xy_d]*len(x), label="XY", c=colors[2], ls=linest[0])
    axc.plot(x, [phi_d] * len(x), label="PHI", c=colors[2], ls=linest[1])
    axc.legend()

    axx = axxy[ci, ri]
    fxy.suptitle("xy-correlation")
    axx.set_title("REP {}".format(rep))
    axx.set_xscale('log')
    axx.plot(x, xyZ_S1, label="bSFA_S", c=colors[0], ls=linest[0])
    axx.plot(x, xyZ_E1, label="bSFA_E", c=colors[0], ls=linest[1])
    axx.plot(x, xyz_S[i], label="incSFA_S", c=colors[1], ls=linest[0])
    axx.plot(x, xyz_E[i], label="incSFA_E", c=colors[1], ls=linest[1])
    axx.legend()

    axp = axphi[ci, ri]
    fphi.suptitle("phi-correlation")
    axp.set_title("REP {}".format(rep))
    axp.set_xscale('log')
    axp.plot(x, phiZ_S1, label="bSFA_S", c=colors[0], ls=linest[0])
    axp.plot(x, phiZ_E1, label="bSFA_E", c=colors[0], ls=linest[1])
    axp.plot(x, phiz_S[i], label="incSFA_S", c=colors[1], ls=linest[0])
    axp.plot(x, phiz_E[i], label="incSFA_E", c=colors[1], ls=linest[1])
    axp.legend()
plt.show()
