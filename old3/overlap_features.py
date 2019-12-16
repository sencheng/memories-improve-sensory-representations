from core import semantic, system_params, input_params, streamlined, sensory
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
import sklearn.linear_model
import pickle

PATH = "/local/results/lromix_o1850t/"
SFA1 = "sfa1.p"
SFA2S = "inc1_eps1_0.sfa"
SFA2E = "inc1_eps1_39.sfa"

LETTER1 = "L"
LETTER2 = "T"
NOI = 0.1

NFEAT = 3

PARAMETERS = system_params.SysParamSet()

normparm = dict(number_of_snippets=1, snippet_length=None, movement_type='sample', movement_params=dict(x_range=None, y_range=None, t_range=None, x_step=0.05, y_step=0.05, t_step=22.5),
            object_code=input_params.make_object_code([LETTER1, LETTER2]), sequence=[0], input_noise=NOI)
parm1 = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER1), sequence=[0], input_noise=NOI)
parm2 = dict(number_of_snippets=1, snippet_length=None, movement_type='copy_traj', movement_params=dict(latent=[[0, 0, 1, 0]], ranges=iter([[0]])),
            object_code=input_params.make_object_code(LETTER2), sequence=[0], input_noise=NOI)

sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

sfa1 = semantic.load_SFA(PATH+SFA1)
sfa2S = semantic.load_SFA(PATH+SFA2S)
sfa2E = semantic.load_SFA(PATH+SFA2E)

normseq, normcat, normlat = sensys.generate(**normparm)
normcat = np.array(normcat)
normlat = np.array(normlat)
norm_target = np.append(normlat, normcat[:,None], axis=1)
norm_y = semantic.exec_SFA(sfa1, normseq)
normalizer1 = streamlined.normalizer(norm_y, PARAMETERS.normalization)
norm_yw = normalizer1(norm_y)
norm_zS = semantic.exec_SFA(sfa2S, norm_yw)
norm_zE = semantic.exec_SFA(sfa2E, norm_yw)
normalizerS = streamlined.normalizer(norm_zS, PARAMETERS.normalization)
normalizerE = streamlined.normalizer(norm_zE, PARAMETERS.normalization)
norm_zSw = normalizerS(norm_zS)
norm_zEw = normalizerS(norm_zE)

sam1 = np.random.rand(2)*2-1
mlx = []
sfa_outS, sfa_outE = [], []
calculatedS, calculatedE = [], []
ml = 0
parm1["movement_params"]["latent"] = [[sam1[0], sam1[1], 1, 0]]
parm1["movement_params"]["ranges"] = iter([[0]])
x_sam1, _, _ = sensys.generate(**parm1)
parm2["movement_params"]["latent"] = [[sam1[0], sam1[1], 1, 0]]
parm2["movement_params"]["ranges"] = iter([[0]])
x_sam2, _, _ = sensys.generate(**parm2)
y_sam1 = semantic.exec_SFA(sfa1, x_sam1)[0]
y_sam2 = semantic.exec_SFA(sfa1, x_sam2)[0]
y_sam1w = normalizer1(y_sam1)
y_sam2w = normalizer1(y_sam2)
z_sam1S = semantic.exec_SFA(sfa2S, y_sam1w)
z_sam2S = semantic.exec_SFA(sfa2S, y_sam2w)
z_sam1Sw = normalizerS(z_sam1S)[:,:NFEAT]
z_sam2Sw = normalizerS(z_sam2S)[:,:NFEAT]
z_sam1E = semantic.exec_SFA(sfa2E, y_sam1w)
z_sam2E = semantic.exec_SFA(sfa2E, y_sam2w)
z_sam1Ew = normalizerE(z_sam1E)[:,:NFEAT]
z_sam2Ew = normalizerE(z_sam2E)[:,:NFEAT]
while ml <= 1:
    x_loc1 = x_sam2 * ml + x_sam1 * (1 - ml)

    y_loc1 = semantic.exec_SFA(sfa1, x_loc1)[0]
    y_loc1w = normalizer1(y_loc1)
    z_loc1S = semantic.exec_SFA(sfa2S, y_loc1w)
    z_loc1Sw = normalizerS(z_loc1S)[:,:NFEAT]
    z_loc1E = semantic.exec_SFA(sfa2E, y_loc1w)
    z_loc1Ew= normalizerE(z_loc1E)[:,:NFEAT]
    sfa_outS.append(z_loc1Sw[0])
    sfa_outE.append(z_loc1Ew[0])

    calculatedS.append(z_sam2Sw[0] * ml + z_sam1Sw[0] * (1 - ml))
    calculatedE.append(z_sam2Ew[0] * ml + z_sam1Ew[0] * (1 - ml))

    mlx.append(ml)
    ml+=0.01

sfa_outS = np.array(sfa_outS)
sfa_outE = np.array(sfa_outE)
calculatedS = np.array(calculatedS)
calculatedE = np.array(calculatedE)

f, ax = plt.subplots(2,1)
ax[0].set_title("S")
ax[1].set_title("E")
for i in range(NFEAT):
    ax[0].plot(mlx, sfa_outS[:,i], label="sfa_out{}".format(i))
    ax[1].plot(mlx, sfa_outE[:, i], label="sfa_out{}".format(i))
    ax[0].plot(mlx, calculatedS[:,i], label="calc{}".format(i))
    ax[1].plot(mlx, calculatedE[:, i], label="calc{}".format(i))
ax[0].legend()
ax[1].legend()
plt.show()