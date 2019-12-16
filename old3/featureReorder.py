import numpy as np
from matplotlib import pyplot as plt
from core import tools, episodic, semantic, streamlined, system_params
import scipy.spatial.distance
import pickle

RESPATH = "../results/"

DRAW = False
# SNLEN_LIST = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
# SNLEN_LIST = [2,3,5,8,12,20,40,100,200,600]
SNLEN_LIST = [2,5,20,100] #forming0, 3, 13, 25
# RETNOI_LIST = [0,1,2,3,4]
RETNOI_LIST = [0,1,2,3,4]
DIMENSION = 288

PARAMETERS = system_params.SysParamSet()

sfa1 = semantic.load_SFA(RESPATH+"reorderN3a/sfadef1train0.sfa")
forming0data = np.load(RESPATH+"reorderN3a/forming0.npz")
forming0seq, forming0cat, forming0lat, forming0ran = forming0data["forming_sequenceX"], forming0data["forming_categories"], forming0data["forming_latent"], forming0data["forming_ranges"]
forming3data = np.load(RESPATH+"reorderN3a/forming3.npz")
forming3seq, forming3cat, forming3lat, forming3ran = forming3data["forming_sequenceX"], forming3data["forming_categories"], forming3data["forming_latent"], forming3data["forming_ranges"]
forming13data = np.load(RESPATH+"reorderN3a/forming13.npz")
forming13seq, forming13cat, forming13lat, forming13ran = forming13data["forming_sequenceX"], forming13data["forming_categories"], forming13data["forming_latent"], forming13data["forming_ranges"]
forming25data = np.load(RESPATH+"reorderN3a/forming25.npz")
forming25seq, forming25cat, forming25lat, forming25ran = forming25data["forming_sequenceX"], forming25data["forming_categories"], forming25data["forming_latent"], forming25data["forming_ranges"]

z0 = semantic.exec_SFA(sfa1, forming0seq)
z3 = semantic.exec_SFA(sfa1, forming3seq)
z13 = semantic.exec_SFA(sfa1, forming13seq)
z25 = semantic.exec_SFA(sfa1, forming25seq)

# f1, ax1 = plt.subplots(2,2)
# ax1[0,0].hist(scipy.spatial.distance.cdist(z0[0:1], z0, "euclidean")[0], bins=50)
# ax1[0,0].set_title("snlen=2")
# ax1[0,1].hist(scipy.spatial.distance.cdist(z3[0:1], z3, "euclidean")[0], bins=50)
# ax1[0,1].set_title("snlen=5")
# ax1[1,0].hist(scipy.spatial.distance.cdist(z13[0:1], z13, "euclidean")[0], bins=50)
# ax1[1,0].set_title("snlen=20")
# ax1[1,1].hist(scipy.spatial.distance.cdist(z25[0:1], z25, "euclidean")[0], bins=50)
# ax1[1,1].set_title("snlen=50")
# f1.suptitle("not normalized")

z0w = streamlined.normalizer(z0, PARAMETERS.normalization)(z0)
z3w = streamlined.normalizer(z3, PARAMETERS.normalization)(z3)
z13w = streamlined.normalizer(z13, PARAMETERS.normalization)(z13)
z25w = streamlined.normalizer(z25, PARAMETERS.normalization)(z25)

# f2, ax2 = plt.subplots(2,2)
# ax2[0,0].hist(scipy.spatial.distance.cdist(z0w[0:1], z0w, "euclidean")[0], bins=50)
# ax2[0,0].set_title("snlen=2")
# ax2[0,1].hist(scipy.spatial.distance.cdist(z3w[0:1], z3w, "euclidean")[0], bins=50)
# ax2[0,1].set_title("snlen=5")
# ax2[1,0].hist(scipy.spatial.distance.cdist(z13w[0:1], z13w, "euclidean")[0], bins=50)
# ax2[1,0].set_title("snlen=20")
# ax2[1,1].hist(scipy.spatial.distance.cdist(z25w[0:1], z25w, "euclidean")[0], bins=50)
# ax2[1,1].set_title("snlen=50")
# f2.suptitle("normalized")
#
# plt.show()

delta_ret, delta_retmem, delta_seq = [], [], []

for snlen in SNLEN_LIST:
    seq = eval("z{}w".format(0 if snlen==2 else 3 if snlen==5 else 13 if snlen==20 else 25))
    delta_ret.append([])
    delta_retmem.append([])
    print("snlen = {}".format(snlen))
    npoints = 30000
    # snlen = 60
    nsnip = npoints//snlen
    retlen = 80
    nret = npoints//retlen
    # retnoi = 0.2

    # dx = 0.05
    # step=5

    # def _clamp(x, b): return np.clip(x, -b, b)
    # def _randN(ds) : return np.random.normal(0, ds, DIMENSION)

    # seq = []
    pats = []
    keys = []
    # for sni in range(nsnip):
    #     xy = 2*np.random.rand(DIMENSION)-1
    #     seq.append(xy)
    #     # pats.append(xy)
    #     for i in range(snlen-1):
    #         xy = _clamp(seq[-1] + dx * _randN(step), 1)  # Adjust coefficient of step size for coverage
    #         seq.append(xy)
    #         # keys.append(xy)
    #         # if i < snlen-2:
    #         #     pats.append(xy)
    #     if DRAW:
    #         temp = np.array(seq[-snlen:])
    #         x = temp[:,0]
    #         y = temp[:,1]
    #         plt.plot(x,y,'bo-')
    # # pats = np.array(pats)
    # # keys = np.array(keys)
    #
    # if DRAW:
    #     plt.show()

    memory = episodic.EpisodicMemory(p_dim=DIMENSION, retrieval_length=retlen, retrieval_noise=0)
    for sni in range(nsnip):
        memory.store_sequence(seq[sni*snlen:sni*snlen+snlen])
        pats.extend(seq[sni*snlen:sni*snlen+snlen-1])
        keys.extend(seq[sni * snlen+1:sni * snlen + snlen])
    pats = np.array(pats)
    keys = np.array(keys)

    # _ = memory.retrieve_sequence(0)
    # plt.hist(np.sqrt(memory.dmat[0]), bins=50)
    # plt.figure()
    # plt.hist(np.sqrt(scipy.spatial.distance.cdist(keys[0:1], pats, 'sqeuclidean')[0]), bins=50)
    # plt.show()

    for retnoi in RETNOI_LIST:
        ret = []
        retmem = []
        for reti in range(nret):
            cue = seq[np.random.randint(len(seq))]
            retmem.extend(memory.retrieve_sequence(cue, ret_noise=retnoi))
            for i in range(retlen):
                if retnoi > 0:
                    noi = np.random.normal(0, retnoi, DIMENSION)
                else:
                    noi = np.array([0.]*DIMENSION)
                p = cue + noi
                dist_mat = np.sum((p-pats)**2, axis=1)
                mindex = np.argmin(dist_mat)
                ret.append(pats[mindex])
                cue = keys[mindex]

        delta_ret[-1].append(tools.delta_diff(ret))
        delta_retmem[-1].append(tools.delta_diff(retmem))

    delta_seq.append(tools.delta_diff(seq))
delta_ret = np.array(delta_ret)
delta_retmem = np.array(delta_retmem)

# plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")
# for ni, nnn in enumerate(RETNOI_LIST):
#     plt.plot(SNLEN_LIST, np.mean(delta_ret[:, ni:ni+1, :], axis=2), ['y-', 'g-', 'm-', 'r-', 'c-'][ni], label="ret {}".format(nnn))
# plt.legend()
# plt.xscale('log')

# plt.figure()
# plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")
# for ni, nnn in enumerate(RETNOI_LIST):
#     plt.plot(SNLEN_LIST, np.mean(delta_retmem[:, ni:ni+1, :], axis=2), ['y-', 'g-', 'm-', 'r-', 'c-'][ni], label="retmem {}".format(nnn))
# plt.legend()
# plt.xscale('log')
#
# plt.show()

np.save("../results/featre2/delta_seq.npy", np.array(delta_seq))
np.save("../results/featre2/delta_ret.npy", delta_ret)
np.save("../results/featre2/delta_retmem.npy", delta_retmem)
# with open("../results/simpre288/memory.p", 'wb') as f:
#     pickle.dump(memory,f)
