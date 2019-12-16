from core import input_params, system_params, sensory, semantic, streamlined, episodic
import numpy as np
from matplotlib import pyplot as plt

PATH = "../results/lro_o1850t/"
SFA1 = "sfa1.p"
SFA2S = "inc1_eps1_0.sfa"
SFA2E = "inc1_eps1_39.sfa"

PARAMETERS = system_params.SysParamSet()

nsnip = 600*25

nshow = 9

PARAMETERS.st2['movement_type'] = 'gaussian_walk'
PARAMETERS.st2['movement_params'] = dict(dx = 0.05, dt = 0.05, step=5)
PARAMETERS.st2['snippet_length'] = 80
PARAMETERS.st2["number_of_snippets"] = nsnip//PARAMETERS.st2['snippet_length']

PARAMETERS.st2['memory']['retrieval_length'] = 80
PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['retrieval_noise'] = 4
PARAMETERS.st2['memory']['depress_params'] = dict(cost=400, recovery_time_constant=400, activation_function='lambda X : X')
PARAMETERS.st2['memory']['smoothing_percentile'] = 100

sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
seq, cat, lat, ranges = sensys.generate(fetch_indices=True, **PARAMETERS.st2)
cat = np.array(cat)
lat = np.array(lat)

sfa1 = semantic.load_SFA(PATH+SFA1)
y = semantic.exec_SFA(sfa1, seq)
yw = streamlined.normalizer(y, PARAMETERS.normalization)(y)

memory = episodic.EpisodicMemory(sfa1[-1].output_dim, **PARAMETERS.st2['memory'])
for ran in ranges:
    liran = list(ran)
    memory.store_sequence(yw[liran], categories=cat[liran], latents=lat[liran])

_ = memory.retrieve_sequence(0)

noilist = [0.2, 1, 2, 3, 4, 5]
cnt = len(noilist)
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))
fn, axn = plt.subplots(rows, cols, squeeze=False, figsize=(19.2, 9.96))
idx = 0
nabs = []
for ni, noi in enumerate(noilist):
    for i in range(nsnip*2):
        nabs.append(np.linalg.norm(np.random.normal(0, noi + 1e-20, memory.helper_dim)))
    axn0 = axn[ni // cols, ni % cols]
    axn0.hist(nabs, bins=50)
    axn0.set_title(noi)
    nabs = []
fn.suptitle("mean = sqrt(288*noi*2)")

plt.savefig("../results/disdis/disdis2_noi.svg")


# def press(event):
#     tit = ax.get_title()
#     if event.key == "right":
#         n = int(tit.split()[0])+1
#         if n == len(memory.dmat[0]):
#             n = 0
#     elif event.key == "left":
#         n = int(tit.split()[0])-1
#         if n == -1:
#             n = len(memory.dmat[0])-1
#     else:
#         return
#     ax.clear()
#     ax.hist(np.sqrt(memory.dmat[n]), bins=50)
#     ax.set_title("{} ({})".format(n, n % (PARAMETERS.st2["snippet_length"]-1)))
#     f.canvas.draw()
#
# f, ax = plt.subplots()
#
# ax.hist(np.sqrt(memory.dmat[idx]), bins=50)
# ax.set_title("{} ({})".format(idx, idx%PARAMETERS.st2["snippet_length"]))
#
# f.canvas.mpl_connect('key_press_event', press)

cnt = nshow
cols = int(np.sqrt(cnt))
rows = int(cnt / cols) + int(bool(cnt % cols))

if nshow < PARAMETERS.st2['snippet_length']:
    loopran = range(PARAMETERS.st2['snippet_length']-nshow, PARAMETERS.st2['snippet_length'])
else:
    loopran = range(nshow)

f, ax = plt.subplots(rows, cols, squeeze=False, sharex=True, figsize=(19.2, 9.96))
for i, iv in enumerate(loopran):
    ax0 = ax[i // cols, i % cols]
    # ax0.hist(np.sqrt(memory.dmat[iv]), bins=50)
    ax0.hist(memory.dmat[iv], bins=50)
    ax0.set_title("{} ({})".format(iv, iv % (PARAMETERS.st2["snippet_length"] - 1)))

plt.savefig("../results/disdis/disdis2_seq{}.svg".format(PARAMETERS.st2['snippet_length']))
# plt.show()