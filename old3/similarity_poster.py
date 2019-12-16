from core import semantic, sensory, system_params, input_params, streamlined

import numpy as np
from matplotlib import pyplot as plt
import itertools

# PATH = "/local/results/lro_o1850t/"
# SFA1FILE = "sfa1.p"
# SFA2SFILE = "inc1_eps1_0.sfa"
# SFA2EFILE = "inc1_eps1_39.sfa"

PATHS = "/local/results/reorder4a/"
PATHE = "/local/results/reorder4a/"
SFA1SFILE = "sfadef1train0.sfa"
SFA1EFILE = "sfadef1train0.sfa"
SFA2SFILE = "res0_sfa2S.sfa"
SFA2EFILE = "res25_sfa2S.sfa"

PARAMETERS = system_params.SysParamSet()

LETTER = 'L'
DIFL = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
THRESHL = [0.99]
# THRESHL = [0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.]
THRESHL2 = list(1-np.array(THRESHL))
FIXDIF = [1,2,3,4,5,6]
N = 2000
COLOR_INTERVAL = 0.7

mcolor_s = 'b'
mshape_s = '+'
mcolor_e = 'b'
mshape_e = 'x'

DRAW_HISTS = False
BIN_SIZE = 0.2

normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=22.5),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=0)
parm = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=0)

if SFA1SFILE is not None:
    sfa1S = semantic.load_SFA(PATHS+SFA1SFILE)
    sfa1E = semantic.load_SFA(PATHE + SFA1EFILE)
sfa2S = semantic.load_SFA(PATHS+SFA2SFILE)
sfa2E = semantic.load_SFA(PATHE+SFA2EFILE)

print("Generating input")
sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

samp, _, _ = sensory_system.generate(**normparm)
print("Executing SFA")
if SFA1SFILE is not None:
    sampyS = semantic.exec_SFA(sfa1S, samp)
    sampyE = semantic.exec_SFA(sfa1E, samp)
    normalizer_yS = streamlined.normalizer(sampyS, PARAMETERS.normalization)
    normalizer_yE = streamlined.normalizer(sampyE, PARAMETERS.normalization)
    sampy_wS = normalizer_yS(sampyS)
    sampy_wE = normalizer_yE(sampyE)
    sampzS = semantic.exec_SFA(sfa2S, sampy_wS)
    sampzE = semantic.exec_SFA(sfa2E, sampy_wE)
    normalizer_zS = streamlined.normalizer(sampzS, PARAMETERS.normalization)
    normalizer_zE = streamlined.normalizer(sampzE, PARAMETERS.normalization)
else:
    sampzS = semantic.exec_SFA(sfa2S, samp)
    sampzE = semantic.exec_SFA(sfa2E, samp)
    normalizer_zS = streamlined.normalizer(sampzS, PARAMETERS.normalization)
    normalizer_zE = streamlined.normalizer(sampzE, PARAMETERS.normalization)

cdf_sameS = []
cdf_sameE = []
cdf_diffS = []
cdf_diffE = []

if DRAW_HISTS:
    cnt = len(DIFL)
    cols = int(np.sqrt(cnt))
    rows = int(cnt / cols) + int(bool(cnt % cols))
    fh, axh = plt.subplots(rows, cols, sharex=True, squeeze=False)

for id, DIF in enumerate(DIFL):
    print("DIF", DIF)
    d_sameS = []
    d_sameE = []
    d_diffS = []
    d_diffE = []
    for i in range(N):
        rr = np.random.rand(2)*2-1
        lat = np.concatenate((rr, [1, 0]))
        rr = np.random.rand(2) * 2 - 1
        lat2 = np.concatenate((rr, [1, 0]))
        while True:
            rn = np.random.normal(0, DIF, 2)
            rn /= np.linalg.norm(rn) / DIF
            dif = np.concatenate((rn, [0, 0]))
            latD = lat+dif
            if np.all(latD >= -1) and np.all(latD <= 1):
                break
        # make random difference vector and normalize to constant length
        parm['movement_params']['ranges'] = iter([[0]])    #  needs to be reset before every call
        parm['movement_params']['latent'] = [list(lat)]
        x0, _, _ = sensory_system.generate(**parm)
        parm['movement_params']['ranges'] = iter([[0]])
        parm['movement_params']['latent'] = [list(latD)]
        x1, _, _ = sensory_system.generate(**parm)
        parm['movement_params']['ranges'] = iter([[0]])
        parm['movement_params']['latent'] = [list(lat2)]
        x2, _, _ = sensory_system.generate(**parm)
        if SFA1SFILE is not None:
            y0S = semantic.exec_SFA(sfa1S, x0)
            y1S = semantic.exec_SFA(sfa1S, x1)
            y2S = semantic.exec_SFA(sfa1S, x2)
            y0_wS = normalizer_yS(y0S)
            y1_wS = normalizer_yS(y1S)
            y2_wS = normalizer_yS(y2S)
            y0E = semantic.exec_SFA(sfa1E, x0)
            y1E = semantic.exec_SFA(sfa1E, x1)
            y2E = semantic.exec_SFA(sfa1E, x2)
            y0_wE = normalizer_yE(y0E)
            y1_wE = normalizer_yE(y1E)
            y2_wE = normalizer_yE(y2E)
            z0S = semantic.exec_SFA(sfa2S, y0_wS)
            z1S = semantic.exec_SFA(sfa2S, y1_wS)
            z2S = semantic.exec_SFA(sfa2S, y2_wS)
            z0E = semantic.exec_SFA(sfa2E, y0_wE)
            z1E = semantic.exec_SFA(sfa2E, y1_wE)
            z2E = semantic.exec_SFA(sfa2E, y2_wE)
        else:
            z0S = semantic.exec_SFA(sfa2S, x0)
            z1S = semantic.exec_SFA(sfa2S, x1)
            z2S = semantic.exec_SFA(sfa2S, x2)
            z0E = semantic.exec_SFA(sfa2E, x0)
            z1E = semantic.exec_SFA(sfa2E, x1)
            z2E = semantic.exec_SFA(sfa2E, x2)
        z0S_w = normalizer_zS(z0S)
        z1S_w = normalizer_zS(z1S)
        z2S_w = normalizer_zS(z2S)
        z0E_w = normalizer_zS(z0E)
        z1E_w = normalizer_zS(z1E)
        z2E_w = normalizer_zS(z2E)
        dsS = np.sqrt(np.sum((z0S_w - z1S_w)**2))
        dsE = np.sqrt(np.sum((z0E_w - z1E_w) ** 2))
        ddS = np.sqrt(np.sum((z0S_w - z2S_w)**2))
        ddE = np.sqrt(np.sum((z0E_w - z2E_w) ** 2))
        d_sameS.append(dsS)
        d_sameE.append(dsE)
        d_diffS.append(ddS)
        d_diffE.append(ddE)

    cdf_sameS.append(np.sort(d_sameS))
    cdf_diffS.append(np.sort(d_diffS))
    cdf_sameE.append(np.sort(d_sameE))
    cdf_diffE.append(np.sort(d_diffE))

    if DRAW_HISTS:
        hiS, binsS, _ = axh[id//cols, id%cols].hist(d_sameS, alpha=0.4, label="simple", bins=int((max(d_sameS) - min(d_sameS))/BIN_SIZE), color='g')
        #xmax = max(np.where(hi > 4)[0])
        # hi, bins, _ = plt.hist(d_diffS, alpha=0.4, label='random', bins=int((max(d_diffS) - min(d_diffS))/BIN_SIZE))
        hiE, binsE, _ = axh[id//cols, id%cols].hist(d_sameE, alpha=0.4, label="episodic", bins=int((max(d_sameE) - min(d_sameE))/BIN_SIZE), color='r')
        bins = binsS if len(binsE) < len(binsS) else binsE
        xmax = max(max(np.where(hiS > 4)[0]), max(np.where(hiE > 4)[0]))
        axh[id // cols, id % cols].set_xlim((0, bins[xmax + 1]))
        # hi, bins, _ = plt.hist(d_diffE, alpha=0.4, label='random', bins=int((max(d_diffE) - min(d_diffE))/BIN_SIZE))
        axh[id // cols, id % cols].set_title("step={}".format(DIF))
        plt.legend()

f0, ax0 = plt.subplots()
ax0.set_xlabel("Difference between feature vectors")
ax0.set_ylabel("Percentage of feature pairs with lower difference")
# ax0.text(1, 1.05, "Classification threshold (%) applied in trials with most similar input pairs")
c_step = COLOR_INTERVAL/len(DIFL)
for i in range(len(DIFL)):
    l1, = ax0.plot(cdf_sameS[i], np.arange(N) / float(N), c=(0.0, 1 - COLOR_INTERVAL + i * c_step, 0.0))
    l2, = ax0.plot(cdf_sameE[i], np.arange(N) / float(N), c=(1 - COLOR_INTERVAL + i * c_step, 0.0, 0.0))
ax0.set_xlim([0,10])
ls, = ax0.plot(-5, 0, c='g', marker=mshape_s, mec=mcolor_s, mfc='none', markersize=12)
le, = ax0.plot(-5, 0, c='r', marker=mshape_e, mec=mcolor_e, mfc='none', markersize=12)
ax0.legend((ls, le), ("simple", "episodic"))
ax0.set_title("CDF")
f0.subplots_adjust(bottom=0.15)

# Learned difference threshold2
cnt = len(THRESHL2)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
f, ax = plt.subplots(rows, cols, sharey=True, sharex=True, squeeze=False)
c_step = COLOR_INTERVAL/len(THRESHL2)
for i, t in enumerate(THRESHL2):
    yy = int(t * N)-1 if t>0 else 0
    xxS = cdf_sameS[-1][yy]
    xxE = cdf_sameE[-1][yy]
    ax0.plot((xxS,xxS), (0, 1), c='g')
    ax0.text(xxS, 1, "{:.2f}".format(t), color='g')
    ax0.plot((xxE, xxE), (0, 1), c='r')
    ax0.text(xxE, 1.02, "{:.2f}".format(t), color='r')
    errS, errE = [], []
    for xlistS, xlistE in zip(cdf_sameS, cdf_sameE):
        errS.append(float(np.searchsorted(xlistS, xxS))/N)
        errE.append(float(np.searchsorted(xlistE, xxE))/N)
    l1, = ax[i//cols, i%cols].plot(DIFL, errS, c='g')
    l2, = ax[i//cols, i % cols].plot(DIFL, errE, c='r')
    ls, = ax0.plot(-5, 0, c='g', marker=mshape_s, mec=mcolor_s, mfc='none', markersize=12)
    le, = ax0.plot(-5, 0, c='r', marker=mshape_e, mec=mcolor_e, mfc='none', markersize=12)
    # ax[i // cols, i % cols].set_title(t)

ax[0,0].set_ylabel("Discrimination error")
ax[0,0].set_xlabel("Distance between input patterns")
ax[0,0].set_xticks([0.025,0.1,0.2])
ax[0,0].legend((ls, le), ("simple", "episodic"))
# f.suptitle("Threshold determined by classification rate2")
f.subplots_adjust(bottom=0.15)

plt.show()
