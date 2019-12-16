from core import semantic, sensory, system_params, input_params, streamlined

import numpy as np
from matplotlib import pyplot as plt
import itertools

PATH = "/local/results/lro_o1850t/"
SFA1FILE = "sfa1.p"
SFA2SFILE = "inc1_eps1_0.sfa"
SFA2EFILE = "inc1_eps1_39.sfa"

PARAMETERS = system_params.SysParamSet()

LETTER = 'L'
DIFL = [0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
THRESHL = [0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.]
THRESHL2 = list(1-np.array(THRESHL))
FIXDIF = [1,2,3,4,5,6]
N = 1000
COLOR_INTERVAL = 0.7

NFEATURES = 3

DRAW_HISTS = True
BIN_SIZE = 0.2

normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=22.5),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=0.1)
parm = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER), sequence=[0], input_noise=0.1)

if SFA1FILE is not None:
    sfa1 = semantic.load_SFA(PATH+SFA1FILE)
sfa2S = semantic.load_SFA(PATH+SFA2SFILE)
sfa2E = semantic.load_SFA(PATH+SFA2EFILE)

print("Generating input")
sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

samp, _, _ = sensory_system.generate(**normparm)
print("Executing SFA")
if SFA1FILE is not None:
    sampy = semantic.exec_SFA(sfa1, samp)
    normalizer_y = streamlined.normalizer(sampy, PARAMETERS.normalization)
    sampy_w = normalizer_y(sampy)
    sampzS = semantic.exec_SFA(sfa2S, sampy_w)
    sampzE = semantic.exec_SFA(sfa2E, sampy_w)
    normalizer_zS = streamlined.normalizer(sampzS[:,:NFEATURES], PARAMETERS.normalization)
    normalizer_zE = streamlined.normalizer(sampzE[:,:NFEATURES], PARAMETERS.normalization)
else:
    sampzS = semantic.exec_SFA(sfa2S, samp)
    sampzE = semantic.exec_SFA(sfa2E, samp)
    normalizer_zS = streamlined.normalizer(sampzS[:,:NFEATURES], PARAMETERS.normalization)
    normalizer_zE = streamlined.normalizer(sampzE[:,:NFEATURES], PARAMETERS.normalization)

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
        lat = np.concatenate((rr, [1,0]))
        rr = np.random.rand(2) * 2 - 1
        lat2 = np.concatenate((rr, [1,0]))
        while True:
            if not DIF == 0:
                rn = np.random.normal(0, DIF, 2)
                rn /= np.linalg.norm(rn) / DIF
                dif = np.concatenate((rn, [0, 0]))
                latD = lat+dif
            else:
                latD = lat
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
        if SFA1FILE is not None:
            y0 = semantic.exec_SFA(sfa1, x0)
            y1 = semantic.exec_SFA(sfa1, x1)
            y2 = semantic.exec_SFA(sfa1, x2)
            y0_w = normalizer_y(y0)
            y1_w = normalizer_y(y1)
            y2_w = normalizer_y(y2)
            z0S = semantic.exec_SFA(sfa2S, y0_w)
            z1S = semantic.exec_SFA(sfa2S, y1_w)
            z2S = semantic.exec_SFA(sfa2S, y2_w)
            z0E = semantic.exec_SFA(sfa2E, y0_w)
            z1E = semantic.exec_SFA(sfa2E, y1_w)
            z2E = semantic.exec_SFA(sfa2E, y2_w)
        else:
            z0S = semantic.exec_SFA(sfa2S, x0)
            z1S = semantic.exec_SFA(sfa2S, x1)
            z2S = semantic.exec_SFA(sfa2S, x2)
            z0E = semantic.exec_SFA(sfa2E, x0)
            z1E = semantic.exec_SFA(sfa2E, x1)
            z2E = semantic.exec_SFA(sfa2E, x2)
        z0S_w = normalizer_zS(z0S[:,:NFEATURES])
        z1S_w = normalizer_zS(z1S[:,:NFEATURES])
        z2S_w = normalizer_zS(z2S[:,:NFEATURES])
        z0E_w = normalizer_zS(z0E[:,:NFEATURES])
        z1E_w = normalizer_zS(z1E[:,:NFEATURES])
        z2E_w = normalizer_zS(z2E[:,:NFEATURES])
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
        bnsS = int((max(d_sameS) - min(d_sameS))/BIN_SIZE)
        if bnsS == 0:
            bnsS = 1
        bnsE = int((max(d_sameE) - min(d_sameE))/BIN_SIZE)
        if bnsE == 0:
            bnsE = 1
        hiS, binsS, _ = axh[id//cols, id%cols].hist(d_sameS, alpha=0.4, label="simple", bins=bnsS, color='b')
        #xmax = max(np.where(hi > 4)[0])
        # hi, bins, _ = plt.hist(d_diffS, alpha=0.4, label='random', bins=int((max(d_diffS) - min(d_diffS))/BIN_SIZE))
        hiE, binsE, _ = axh[id//cols, id%cols].hist(d_sameE, alpha=0.4, label="episodic", bins=bnsE, color='r')
        bins = binsS if len(binsE) < len(binsS) else binsE
        xmax = max(max(np.where(hiS > 4)[0]), max(np.where(hiE > 4)[0]))
        axh[id // cols, id % cols].set_xlim((0, bins[xmax + 1]))
        # hi, bins, _ = plt.hist(d_diffE, alpha=0.4, label='random', bins=int((max(d_diffE) - min(d_diffE))/BIN_SIZE))
        axh[id // cols, id % cols].set_title("step={}".format(DIF))
        plt.legend()

f0, ax0 = plt.subplots(1, 2, squeeze=True, sharey=True)
ax0[0].set_xlabel("Difference between feature vectors")
ax0[0].set_ylabel("Percentage of pairs classified as equal")
ax0[0].text(1, 1.05, "Classification threshold (%) applied in trials with most similar input pairs")
c_step = COLOR_INTERVAL/len(DIFL)
for i in range(len(DIFL)):
    l1, = ax0[0].plot(cdf_sameS[i], np.arange(N)/float(N), c=(0.0, 0.0, 1-COLOR_INTERVAL+i*c_step))
    l2, = ax0[0].plot(cdf_sameE[i], np.arange(N)/float(N), c=(1-COLOR_INTERVAL+i*c_step, 0.0, 0.0))
    l1, = ax0[1].plot(cdf_sameS[i], np.arange(N) / float(N), c=(0.0, 0.0, 1 - COLOR_INTERVAL + i * c_step))
    l2, = ax0[1].plot(cdf_sameE[i], np.arange(N) / float(N), c=(1 - COLOR_INTERVAL + i * c_step, 0.0, 0.0))
f0.legend((l1, l2), ("simple", "episodic"))

# Learned difference threshold
cnt = len(THRESHL)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
for offs in range(3):
    f, ax = plt.subplots(rows, cols, sharey=True, sharex=True, squeeze=False)
    c_step = COLOR_INTERVAL/len(THRESHL)
    for i, t in enumerate(THRESHL):
        yy = int(t * N) - 1 if t > 0 else 0
        xxS = cdf_sameS[offs][yy]
        xxE = cdf_sameE[offs][yy]
        # ax0[0].plot((xxS,xxS), (0, 1), c=(0.0, 0.0, 1-COLOR_INTERVAL+i*c_step))
        # ax0[0].text(xxS, 1, "{:.0f}".format(t*100), color=(0.0, 0.0, 1-COLOR_INTERVAL+i*c_step))
        # ax0[0].plot((xxE, xxE), (0, 1), c=(1-COLOR_INTERVAL+i*c_step, 0.0, 0.0))
        # ax0[0].text(xxE, 1.02, "{:.0f}".format(t*100), color=(1-COLOR_INTERVAL+i*c_step, 0.0, 0.0))
        errS, errE = [], []
        for xlistS, xlistE in zip(cdf_sameS, cdf_sameE):
            errS.append(float(np.searchsorted(xlistS, xxS))/N)
            errE.append(float(np.searchsorted(xlistE, xxE))/N)
        l1, = ax[i//cols, i%cols].plot(DIFL, errS, c='b')
        l2, = ax[i//cols, i % cols].plot(DIFL, errE, c='r')
        ax[i // cols, i % cols].set_title(t)

    f.text(0.09, 0.5, "Error rate", rotation=90, horizontalalignment="center")
    f.text(0.5, 0.06, "Step", horizontalalignment="center")
    f.legend((l1, l2), ("simple", "episodic"))
    f.suptitle("Threshold determined by classification rate on DIF={}".format(DIFL[offs]))

# Learned difference threshold2
cnt = len(THRESHL2)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
for offs in range(3):
    f, ax = plt.subplots(rows, cols, sharey=True, sharex=True, squeeze=False)
    c_step = COLOR_INTERVAL/len(THRESHL2)
    for i, t in enumerate(THRESHL2):
        yy = int(t * N)-1 if t>0 else 0
        xxS = cdf_sameS[-1-offs][yy]
        xxE = cdf_sameE[-1-offs][yy]
        # ax0[1].plot((xxS,xxS), (0, 1), c=(0.0, 0.0, 1-COLOR_INTERVAL+(cnt-i)*c_step))
        # ax0[1].text(xxS, 1, "{:.0f}".format(t*100), color=(0.0, 0.0, 1-COLOR_INTERVAL+(cnt-i)*c_step))
        # ax0[1].plot((xxE, xxE), (0, 1), c=(1-COLOR_INTERVAL+(cnt-i)*c_step, 0.0, 0.0))
        # ax0[1].text(xxE, 1.02, "{:.0f}".format(t*100), color=(1-COLOR_INTERVAL+(cnt-i)*c_step, 0.0, 0.0))
        errS, errE = [], []
        for xlistS, xlistE in zip(cdf_sameS, cdf_sameE):
            errS.append(float(np.searchsorted(xlistS, xxS))/N)
            errE.append(float(np.searchsorted(xlistE, xxE))/N)
        l1, = ax[i//cols, i%cols].plot(DIFL, errS, c='b')
        l2, = ax[i//cols, i % cols].plot(DIFL, errE, c='r')
        ax[i // cols, i % cols].set_title(t)

    f.text(0.09, 0.5, "Error rate", rotation=90, horizontalalignment="center")
    f.text(0.5, 0.06, "Step", horizontalalignment="center")
    f.legend((l1, l2), ("simple", "episodic"))
    f.suptitle("Threshold determined by classification rate2 on DIF={}".format(DIFL[-1-offs]))

# Fixed difference threshold
cnt = len(FIXDIF)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
f, ax = plt.subplots(rows, cols, sharey=True, sharex=True, squeeze=False)
c_step = COLOR_INTERVAL/len(FIXDIF)
for i, xx in enumerate(FIXDIF):
    errS, errE = [], []
    for xlistS, xlistE in zip(cdf_sameS, cdf_sameE):
        errS.append(float(np.searchsorted(xlistS, xx))/N)
        errE.append(float(np.searchsorted(xlistE, xx))/N)
    l1, = ax[i//cols, i%cols].plot(DIFL, errS, c='b')
    l2, = ax[i//cols, i % cols].plot(DIFL, errE, c='r')
    ax[i // cols, i % cols].set_title(xx)

f.text(0.09, 0.5, "Error rate", rotation=90, horizontalalignment="center")
f.text(0.5, 0.06, "Step", horizontalalignment="center")
f.legend((l1, l2), ("simple", "episodic"))
f.suptitle("Threshold determined by fixed difference of feature vectors")

plt.show()
