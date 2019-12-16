import numpy as np
from matplotlib import pyplot as plt

PATH = "/local/Sciebo/arbeit/Semantic-Episodic/18-03-01 results/paper figures/"
inds = [522, 101000, 202020]

dataL = np.load("/local/results/inp_sample/inp_sampleL.npz")
dataT = np.load("/local/results/inp_sample/inp_sampleT.npz")

seqL = dataL["sample_sequence"]
seqT = dataT["sample_sequence"]

for ind in inds:
    plt.imshow(np.reshape(seqL[ind], (30,30)), interpolation="none", cmap="Greys")
    plt.axis('off')
    plt.savefig(PATH+"L{}.svg".format(ind), bbox_inches='tight', dpi=150)
    plt.imshow(np.reshape(seqT[ind],(30,30)), interpolation="none", cmap="Greys")
    plt.axis('off')
    plt.savefig(PATH+"T{}.svg".format(ind), bbox_inches='tight', dpi=150)

f0, ax0 = plt.subplots(2,3, figsize=(20,14))
for li, let in enumerate(['L', 'T']):
    for ii, ind in enumerate(inds):
        ax = ax0[li, ii]
        ax.imshow(np.reshape(eval("seq{}".format(let))[ind], (30,30)), interpolation="none", cmap="Greys")
        ax.axis('off')
plt.savefig(PATH+"inputs.svg", bbox_inches='tight')
