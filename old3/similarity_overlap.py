from core import semantic, system_params, input_params, streamlined, sensory, tools
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
import sklearn.linear_model
import pickle

def addnoi(frame, noise_std):
    clip_min = 0
    clip_max = 1
    ret = np.array(frame)
    if noise_std > 0:
        ret += np.random.normal(0, noise_std, np.shape(frame))
    ret = np.clip(ret, clip_min, clip_max)
    return ret

PATHS = "/local/results/replay_o18/"
PATHE = "/local/results/replay_o18/"
SFA1S = "sfa1.p"
SFA1E = "sfa1.p"
SFA2S = "inc1_eps1_0.sfa"
SFA2E = "inc1_eps1_39.sfa"
WHITENERE = "whitener.p"
WHITENERS = "whitener.p"

# PATHS = "/local/results/reorder4a/"
# PATHE = "/local/results/reorder4a/"
# SFA1S = "sfadef1train0.sfa"
# SFA1E = "sfadef1train0.sfa"
# SFA2S = "res0_sfa2E.sfa"
# SFA2E = "res0_sfa2S.sfa"

# PATHS = "/local/results/reorderN4a/"
# PATHE = "/local/results/reorder0a/"
# SFA1S = "sfadef1train0.sfa"
# SFA1E = "sfadef1train0.sfa"
# SFA2S = "res0_sfa2E.sfa"
# SFA2E = "res0_sfa2E.sfa"

mcolor_s = 'b'
mcolor_e = 'b'
# mshape_s = '^'
# mshape_e = 'v'
mshape_s = '+'
mshape_e = 'x'

# MIXING_LEVELS = [0.5, 0.49, 0.475, 0.45, 0.4, 0.3, 0.2, 0.1, 0]
MIXING_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.475, 0.49, 0.5]
# MIXING_LEVELS = [0.5, 0.499, 0.4975, 0.495, 0.49]
# MIXING_LEVELS = [1]

LETTER1 = "T"
LETTER2 = "L"
NOI = 0.1

N=2000
NFEAT = 16

# Whitening settings
# of the data
WHITENER = True
NORM = False
# of the analysis after sfahi
FEATNORM = False

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 12
matplotlib.rcParams['lines.markeredgewidth'] = 2
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

PARAMETERS = system_params.SysParamSet()

normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=22.5),
            object_code=input_params.make_object_code([LETTER1, LETTER2]), sequence=[0], input_noise=NOI)
parm1 = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER1), sequence=[0], input_noise=0)
parm2 = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER2), sequence=[0], input_noise=0)

sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

sfa1S = semantic.load_SFA(PATHS+SFA1S)
sfa1E = semantic.load_SFA(PATHE+SFA1E)
sfa2S = semantic.load_SFA(PATHS+SFA2S)
sfa2E = semantic.load_SFA(PATHE+SFA2E)

if WHITENER:
    with open(PATHS + WHITENERS, 'rb') as w:
        whitenerS = pickle.load(w)
    with open(PATHS + WHITENERE, 'rb') as w2:
        whitenerE = pickle.load(w2)

normseq, normcat, normlat = sensys.generate(**normparm)
normcat = np.array(normcat)
normlat = np.array(normlat)
norm_target = np.append(normlat, normcat[:,None], axis=1)
norm_yS = semantic.exec_SFA(sfa1S, normseq)
norm_yE = semantic.exec_SFA(sfa1E, normseq)
normalizer1S = streamlined.normalizer(norm_yS, PARAMETERS.normalization)
normalizer1E = streamlined.normalizer(norm_yE, PARAMETERS.normalization)
if WHITENER:
    normalizer1S = whitenerS
    normalizer1E = whitenerE
norm_ywS = normalizer1S(norm_yS)
norm_ywE = normalizer1E(norm_yE)
norm_zS = semantic.exec_SFA(sfa2S, norm_ywS)
norm_zE = semantic.exec_SFA(sfa2E, norm_ywE)
normalizerS = streamlined.normalizer(norm_zS, PARAMETERS.normalization)
normalizerE = streamlined.normalizer(norm_zE, PARAMETERS.normalization)
norm_zSw = normalizerS(norm_zS)[:,:NFEAT]
norm_zEw = normalizerS(norm_zE)[:,:NFEAT]

regS = sklearn.linear_model.LinearRegression()
regE = sklearn.linear_model.LinearRegression()
regS.fit(norm_zSw, norm_target)
regE.fit(norm_zEw, norm_target)

latL = []
outS = []
outE = []
hitratesS, hitratesE = [], []
rS, rE, cS, cE = [], [], [], []
decisionsS, decisionsE = [], []

IMS = []
for i in range(N):
    if (i+1)%100 == 0:
        print(i+1)
    sam1 = np.random.rand(2)*2-1
    for iml, ml in enumerate(MIXING_LEVELS):
        if i == 0:
            decisionsS.append([])
            decisionsE.append([])

        parm1["movement_params"]["latent"] = [[sam1[0], sam1[1], 1, 0]]
        parm1["movement_params"]["ranges"] = iter([[0]])
        x_sam1, _, _ = sensys.generate(**parm1)
        parm2["movement_params"]["latent"] = [[sam1[0], sam1[1], 1, 0]]
        parm2["movement_params"]["ranges"] = iter([[0]])
        x_sam2, _, _ = sensys.generate(**parm2)

        x_loc2 = x_sam1 * ml + x_sam2 * (1 - ml)
        x_loc1 = x_sam2 * ml + x_sam1 * (1 - ml)

        x_sam1 = addnoi(x_sam1, NOI)
        x_loc1 = addnoi(x_loc1, NOI)
        x_loc2 = addnoi(x_loc2, NOI)

        IMS.append([x_loc1, x_loc2])

        # if i == 0:
        #     plt.figure()
        #     plt.suptitle("mixing level = {}".format(ml))
        #     plt.subplot(1,4,1)
        #     plt.imshow(np.reshape(x_sam1,(30,30)), interpolation="none", cmap="Greys")
        #     plt.subplot(1, 4, 2)
        #     plt.imshow(np.reshape(x_loc1, (30, 30)), interpolation="none", cmap="Greys")
        #     plt.subplot(1, 4, 3)
        #     plt.imshow(np.reshape(x_loc2, (30, 30)), interpolation="none", cmap="Greys")
        #     plt.subplot(1, 4, 4)
        #     plt.imshow(np.reshape(x_sam2, (30, 30)), interpolation="none", cmap="Greys")
        #     plt.show()

        y_loc1S = semantic.exec_SFA(sfa1S, x_loc1)[0]
        y_loc2S = semantic.exec_SFA(sfa1S, x_loc2)[0]
        y_sam1S = semantic.exec_SFA(sfa1S, x_sam1)[0]
        y_loc1wS = normalizer1S(y_loc1S)
        y_loc2wS = normalizer1S(y_loc2S)
        y_sam1wS = normalizer1S(y_sam1S)
        y_loc1E = semantic.exec_SFA(sfa1E, x_loc1)[0]
        y_loc2E = semantic.exec_SFA(sfa1E, x_loc2)[0]
        y_sam1E = semantic.exec_SFA(sfa1E, x_sam1)[0]
        y_loc1wE = normalizer1E(y_loc1E)
        y_loc2wE = normalizer1E(y_loc2E)
        y_sam1wE = normalizer1E(y_sam1E)
        z_loc1S = semantic.exec_SFA(sfa2S, y_loc1wS)
        z_loc2S = semantic.exec_SFA(sfa2S, y_loc2wS)
        z_sam1S = semantic.exec_SFA(sfa2S, y_sam1wS)
        z_loc1Sw = normalizerS(z_loc1S)[:,:NFEAT]
        z_loc2Sw = normalizerS(z_loc2S)[:,:NFEAT]
        z_sam1Sw = normalizerS(z_sam1S)[:,:NFEAT]
        z_loc1E = semantic.exec_SFA(sfa2E, y_loc1wE)
        z_loc2E = semantic.exec_SFA(sfa2E, y_loc2wE)
        z_sam1E = semantic.exec_SFA(sfa2E, y_sam1wE)
        z_loc1Ew = normalizerE(z_loc1E)[:,:NFEAT]
        z_loc2Ew = normalizerE(z_loc2E)[:,:NFEAT]
        z_sam1Ew = normalizerE(z_sam1E)[:,:NFEAT]
        #distance from sam1 to loc1 is smaller than from sam2 to loc1!
        d1S = np.linalg.norm(z_sam1S-z_loc1S)
        d2S = np.linalg.norm(z_sam1S - z_loc2S)
        d1E = np.linalg.norm(z_sam1E - z_loc1E)
        d2E = np.linalg.norm(z_sam1E - z_loc2E)
        decisionsS[iml].append(1 if d1S<d2S else 2 if d2S<d1S else 0)
        decisionsE[iml].append(1 if d1E < d2E else 2 if d2E < d1E else 0)
        # plt.subplot(1,2,1)
        # plt.imshow(np.reshape(x_loc1, (30,30)),interpolation="none", cmap="Greys")
        # plt.title("({:.3f}, {:.3f})".format(loc1[0], loc1[1]))
        # plt.subplot(1,2,2)
        # plt.imshow(np.reshape(x_loc2, (30, 30)), interpolation="none", cmap="Greys")
        # plt.title("({:.3f}, {:.3f})".format(loc2[0], loc2[1]))
        # plt.show()
        latL.append(sam1)
        outS.append(z_sam1Sw[0])
        outE.append(z_sam1Ew[0])

    ff, axax = plt.subplots(2, len(IMS))
    for imi, iml in enumerate(IMS):
        axax[0,imi].imshow(np.reshape(iml[0], (30,30)),interpolation="none", cmap="Greys")
        axax[0, imi].axis('off')
        axax[1, imi].imshow(np.reshape(iml[1], (30, 30)), interpolation="none", cmap="Greys")
        axax[1, imi].set_title(MIXING_LEVELS[imi])
        axax[1, imi].axis('off')
        ttl = axax[1, imi].title
        ttl.set_position([.5, 1.12])
        if not imi:
            ff.text(0.11, 0.5, "mixing\nlevel", horizontalalignment="right", verticalalignment="center")
            ff.text(0.11, 0.3, "T2[-]", horizontalalignment="right", verticalalignment="center")
            ff.text(0.11, 0.7, "T1[+]", horizontalalignment="right", verticalalignment="center")
    plt.show()
    IMS = []
for iml, ml in enumerate(MIXING_LEVELS):
    arrS = np.array(decisionsS[iml])
    arrE = np.array(decisionsE[iml])
    hitratesS.append((len(np.where(arrS == 1)[0]) + len(np.where(arrS == 0)[0]) // 2) / N)
    hitratesE.append((len(np.where(arrE == 1)[0]) + len(np.where(arrE == 0)[0]) // 2) / N)

lats = np.array(latL)
predictionS = regS.predict(outS)
predictionE = regE.predict(outE)
_, _, r_valueXS, _, _ = scipy.stats.linregress(lats[:, 0], predictionS[:, 0])
_, _, r_valueYS, _, _ = scipy.stats.linregress(lats[:, 1], predictionS[:, 1])
_, _, r_valueXE, _, _ = scipy.stats.linregress(lats[:, 0], predictionE[:, 0])
_, _, r_valueYE, _, _ = scipy.stats.linregress(lats[:, 1], predictionE[:, 1])
rS.append(np.mean((r_valueXS, r_valueYS)))
rE.append(np.mean((r_valueXE, r_valueYE)))
lat_trans = np.transpose(lats)
feat_transS = np.transpose(outS)
feat_transE = np.transpose(outE)
corS = np.corrcoef(np.append(feat_transS,lat_trans, axis=0))[NFEAT:,:-2]
corE = np.corrcoef(np.append(feat_transE, lat_trans, axis=0))[NFEAT:,:-2]
cS.append(np.mean((np.max(np.abs(corS[0, :])), np.max(np.abs(corS[1, :])))))
cE.append(np.mean((np.max(np.abs(corE[0, :])), np.max(np.abs(corE[1, :])))))

f, ax = plt.subplots(1,1, sharex=True, sharey=True, squeeze=False)

axx = ax[0,0]
axx.plot(MIXING_LEVELS, hitratesS, 'g-')
# axx = ax[0,1]
axx.plot(MIXING_LEVELS, hitratesE, 'r-')
# axx.plot([0,0.5], [0.5, 0.5], 'k--')
f.text(0.5, 0.04, 'mixing level', ha='center', va='center', fontdict=font)
f.text(0.06, 0.5, 'hitrate', ha='center', va='center', rotation='vertical', fontdict=font)

limix = ax[0,0].get_xlim()
ls, = ax[0,0].plot(-5, 0, c='g', marker=mshape_s, mec=mcolor_s, mfc='none', markersize=12)
le, = ax[0,0].plot(-5, 0, c='r', marker=mshape_e, mec=mcolor_e, mfc='none', markersize=12)
lrand, = ax[0,0].plot(MIXING_LEVELS, [0.5]*len(MIXING_LEVELS), 'k--')
ax[0,0].set_xlim(limix)
ax[0,0].set_ylim([0.4, 1.05])
# ax[0,0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

f.legend((ls, le, lrand), ("$\epsilon$=0", "$\epsilon$=4", "random"), 1)
# f.suptitle("Hitrate")

plt.show()