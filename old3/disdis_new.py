from core import input_params, system_params, sensory, semantic, streamlined, episodic
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance

PATH = "/local/results/reorderDa/"
SFA1 = "sfadef1train0.sfa"
SFA2S = "inc1_eps1_0.sfa"
SFA2E = "inc1_eps1_39.sfa"

PARAMETERS = system_params.SysParamSet()

nsnip = 600*25

snlen = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]

nshow = 9

for snli, snl in enumerate(snlen):
    forming_data = np.load(PATH+"forming{}.npz".format(snli))
    form_seq, form_cat, form_lat, form_ran = forming_data['forming_sequenceX'], forming_data['forming_categories'], forming_data['forming_latent'], forming_data['forming_ranges']

    sfa1 = semantic.load_SFA(PATH+SFA1)

    form_y = semantic.exec_SFA(sfa1, form_seq)
    form_y_w = streamlined.normalizer(form_y, PARAMETERS.normalization)(form_y)

    dis0 = scipy.spatial.distance.cdist(form_y_w[0:1], form_y_w, 'euclidean')[0]

    sortin0 = np.argsort(dis0)[1:11]
    sortdis0 = dis0[sortin0]

    indices_in_seq = list(range(snl))

    in_seq, out_seq = [], []
    for ind, dis in zip(sortin0, sortdis0):
        if ind in indices_in_seq:
            in_seq.append(dis)
        else:
            out_seq.append(dis)

    plt.figure()
    plt.plot(in_seq, [1] * len(in_seq), 'gx', label="in")
    plt.plot(out_seq, [1]*len(out_seq), 'rx', label="out")
    plt.title("snlen={}".format(snl))
    plt.legend()

plt.show()