from core import semantic, system_params, input_params, streamlined, sensory
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
import sklearn.linear_model
import pickle

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
# SFA2S = "res0_sfa2S.sfa"
# SFA2E = "res0_sfa2E.sfa"

# PATHS = "/local/results/reorderN4a/"
# PATHE = "/local/results/reorder0a/"
# SFA1S = "sfadef1train0.sfa"
# SFA1E = "sfadef1train0.sfa"
# SFA2S = "res0_sfa2E.sfa"
# SFA2E = "res0_sfa2E.sfa"

mcolor_s = 'b'
mshape_s = '+'
mcolor_e = 'b'
mshape_e = 'x'

# SAMPLE_DISTANCES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
SAMPLE_DISTANCES = [1.0]
# MIXING_LEVELS = [0.5, 0.49, 0.475, 0.45, 0.4, 0.3, 0.2, 0.1, 0]
MIXING_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.475, 0.49, 0.5]
# MIXING_LEVELS = [1]

LETTER = "T"
NOI = 0.1

N=2000

# Whitening settings
# of the data
WHITENER = True
NORM = False
# of the analysis after sfahi
FEATNORM = False

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 12
matplotlib.rcParams['lines.markeredgewidth'] = 2
font = {'family' : 'Sans',
        'size'   : 22}

matplotlib.rc('font', **font)

PARAMETERS = system_params.SysParamSet()

normparm_alt = dict(number_of_snippets=100, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=NOI)
normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=22.5),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=NOI)
parm = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=NOI)

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
norm_zSw = normalizerS(norm_zS)
norm_zEw = normalizerS(norm_zE)

regS = sklearn.linear_model.LinearRegression()
regE = sklearn.linear_model.LinearRegression()
regS.fit(norm_zSw, norm_target)
regE.fit(norm_zEw, norm_target)

samp_cnt = len(SAMPLE_DISTANCES)
latL = []
outS = []
outE = []
hitratesS, hitratesE = [], []
rS, rE, cS, cE = [], [], [], []
IMS=[]
for isd, sd in enumerate(SAMPLE_DISTANCES):
    print(sd)
    decisionsS, decisionsE = [], []
    latL.append([])
    outS.append([])
    outE.append([])
    hitratesS.append([])
    hitratesE.append([])
    for i in range(N):
        if (i+1)%100 == 0:
            print(i+1)
        while True:
            sam1 = np.random.rand(2)*2-1
            if not sd == 0:
                offs = np.random.rand(2)*2-1
                offs /= np.linalg.norm(offs) / sd
                sam2 = sam1+offs
            else:
                sam2 = sam1
            if min(sam2) > -1 and max(sam2) < 1:
                break
        mixing_vector = sam2-sam1
        for iml, ml in enumerate(MIXING_LEVELS):
            if i == 0:
                decisionsS.append([])
                decisionsE.append([])
            loc1 = sam1+ml*mixing_vector
            loc2 = sam2-ml*mixing_vector
            parm["movement_params"]["latent"] = [[loc1[0], loc1[1], 1, 0]]
            parm["movement_params"]["ranges"] = iter([[0]])
            x_loc1, _, _ = sensys.generate(**parm)
            parm["movement_params"]["latent"] = [[loc2[0], loc2[1], 1, 0]]
            parm["movement_params"]["ranges"] = iter([[0]])
            x_loc2, _, _ = sensys.generate(**parm)
            parm["movement_params"]["latent"] = [[sam1[0], sam1[1], 1, 0]]
            parm["movement_params"]["ranges"] = iter([[0]])
            x_sam1, _, _ = sensys.generate(**parm)

            IMS.append([x_loc1, x_loc2])

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
            z_loc1Sw = normalizerS(z_loc1S)
            z_loc2Sw = normalizerS(z_loc2S)
            z_sam1Sw = normalizerS(z_sam1S)
            z_loc1E = semantic.exec_SFA(sfa2E, y_loc1wE)
            z_loc2E = semantic.exec_SFA(sfa2E, y_loc2wE)
            z_sam1E = semantic.exec_SFA(sfa2E, y_sam1wE)
            z_loc1Ew = normalizerE(z_loc1E)
            z_loc2Ew = normalizerE(z_loc2E)
            z_sam1Ew = normalizerE(z_sam1E)
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
        hitratesS[isd].append((len(np.where(arrS == 1)[0]) + len(np.where(arrS == 0)[0]) // 2) / N)
        hitratesE[isd].append((len(np.where(arrE == 1)[0]) + len(np.where(arrE == 0)[0]) // 2) / N)
        latL[isd].extend([loc1, loc2])
        outS[isd].extend([z_sam1Sw[0], z_loc1Sw[0]])
        outE[isd].extend([z_sam1Ew[0], z_loc1Ew[0]])

    lats = np.array(latL[isd])
    predictionS = regS.predict(outS[isd])
    predictionE = regE.predict(outE[isd])
    _, _, r_valueXS, _, _ = scipy.stats.linregress(lats[:, 0], predictionS[:, 0])
    _, _, r_valueYS, _, _ = scipy.stats.linregress(lats[:, 1], predictionS[:, 1])
    _, _, r_valueXE, _, _ = scipy.stats.linregress(lats[:, 0], predictionE[:, 0])
    _, _, r_valueYE, _, _ = scipy.stats.linregress(lats[:, 1], predictionE[:, 1])
    rS.append(np.mean((r_valueXS, r_valueYS)))
    rE.append(np.mean((r_valueXE, r_valueYE)))
    lat_trans = np.transpose(lats)
    feat_transS = np.transpose(outS[isd])
    feat_transE = np.transpose(outE[isd])
    corS = np.corrcoef(np.append(feat_transS,lat_trans, axis=0))[16:,:-2]
    corE = np.corrcoef(np.append(feat_transE, lat_trans, axis=0))[16:,:-2]
    cS.append(np.mean((np.max(np.abs(corS[0, :])), np.max(np.abs(corS[1, :])))))
    cE.append(np.mean((np.max(np.abs(corE[0, :])), np.max(np.abs(corE[1, :])))))

    print("=================")

cols = int(np.sqrt(samp_cnt))
rows = int(samp_cnt / cols) + int(bool(samp_cnt % cols))
f, ax = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)

for isd, dsd in enumerate(SAMPLE_DISTANCES):
    axx = ax[isd // cols, isd % cols]
    axx.plot(MIXING_LEVELS, hitratesS[isd], 'g-')
    axx.plot(MIXING_LEVELS, hitratesE[isd], 'r-')
    axx.plot([0,0.5], [0.5, 0.5], 'k--')
    axx.set_title("sample distance = {}".format(dsd))
f.text(0.5, 0.04, 'mixing level', ha='center', va='center', fontdict=font)
f.text(0.06, 0.5, 'hitrate', ha='center', va='center', rotation='vertical', fontdict=font)

limix = ax[0,0].get_xlim()
ls, = ax[0,0].plot(-5, 0, c='g', marker=mshape_s, mec=mcolor_s, mfc='none', markersize=12)
le, = ax[0,0].plot(-5, 0, c='r', marker=mshape_e, mec=mcolor_e, mfc='none', markersize=12)
lrand, = ax[0,0].plot(-5, 0, 'k--')
ax[0,0].set_xlim(limix)
ax[0,0].set_ylim([0.4, 1.05])
# ax[0,0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

f.legend((ls, le, lrand), ("$\epsilon$=0", "$\epsilon$=4", "random"), 1)
# f.suptitle("Hitrate")

plt.show()