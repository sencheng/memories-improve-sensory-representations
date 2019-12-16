from core import semantic, sensory, system_params, input_params, streamlined

import numpy as np
from matplotlib import pyplot as plt
import itertools

PATH = "/local/results/learnrate2/"
SFA1FILE = "sfa1.p"
SFA2SFILE = "inc1_eps1_0.sfa"
SFA2EFILE = "inc1_eps1_39.sfa"

PARAMETERS = system_params.SysParamSet()

DIFL = [0.1, 0.2, 0.5]
NOIL = [0.1, 0.2, 0.5]

BIN_SIZE = 0.2

for NOI in NOIL:
    normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=45),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=NOI)
    parm = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=NOI)

    sfa1 = semantic.load_SFA(PATH+SFA1FILE)
    sfa2S = semantic.load_SFA(PATH+SFA2SFILE)
    sfa2E = semantic.load_SFA(PATH+SFA2EFILE)

    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

    samp, _, _ = sensory_system.generate(**normparm)
    sampy = semantic.exec_SFA(sfa1, samp)
    normalizer_y = streamlined.normalizer(sampy, PARAMETERS.normalization)
    sampy_w = normalizer_y(sampy)
    sampzS = semantic.exec_SFA(sfa2S, sampy_w)
    sampzE = semantic.exec_SFA(sfa2E, sampy_w)
    normalizer_zS = streamlined.normalizer(sampzS, PARAMETERS.normalization)
    normalizer_zE = streamlined.normalizer(sampzE, PARAMETERS.normalization)

    for DIF in DIFL:
        d_sameS = []
        d_sameE = []
        d_diffS = []
        d_diffE = []
        for i in range(500):
            lat = np.random.rand(4)*2-1
            dif = np.random.normal(0, DIF, 4)
            dif /= np.linalg.norm(dif)/DIF
            # make random difference vector and normalize to constant length
            parm['movement_params']['ranges'] = iter([[0]])    #  needs to be reset before every call
            parm['movement_params']['latent'] = [list(lat)]
            x0, _, _ = sensory_system.generate(**parm)
            parm['movement_params']['ranges'] = iter([[0]])
            x1, _, _ = sensory_system.generate(**parm)
            parm['movement_params']['ranges'] = iter([[0]])
            parm['movement_params']['latent'] = [list(lat+dif)]
            x2, _, _ = sensory_system.generate(**parm)
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

        plt.figure()
        plt.subplot(2,1,1)
        plt.title("SIMPLE")
        hi, bins, _ = plt.hist(d_sameS, alpha=0.4, label='same', bins=int((max(d_sameS) - min(d_sameS))/BIN_SIZE))
        xmax = max(np.where(hi > 4)[0])
        plt.xlim((0, bins[xmax+1]))
        hi, bins, _ = plt.hist(d_diffS, alpha=0.4, label='diff', bins=int((max(d_diffS) - min(d_diffS))/BIN_SIZE))
        xmax = max(np.where(hi > 4)[0])
        plt.xlim((0, bins[xmax+1]))
        plt.subplot(2,1,2)
        plt.title("EPISODIC")
        hi, bins, _ = plt.hist(d_sameE, alpha=0.4, label='same', bins=int((max(d_sameE) - min(d_sameE))/BIN_SIZE))
        xmax = max(np.where(hi > 4)[0])
        plt.xlim((0, bins[xmax+1]))
        hi, bins, _ = plt.hist(d_diffE, alpha=0.4, label='diff', bins=int((max(d_diffE) - min(d_diffE))/BIN_SIZE))
        xmax = max(np.where(hi > 4)[0])
        plt.xlim((0, bins[xmax+1]))
        plt.legend()
        plt.suptitle("DIF={}, NOI={}".format(DIF, NOI))
plt.show()