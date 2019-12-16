import numpy as np
from matplotlib import pyplot as plt
from core import tools
from core import episodic, semantic, streamlined, system_params
import scipy.spatial.distance
import pickle

LOADPATH = "../results/reorder4a/"
PATH = "../results/flauschRand288xyNoOpt/"

DRAW = False
# SNLEN_LIST = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,60,70,80,90,100,120,150,200,300,600]
# SNLEN_LIST = [2,3,5,8,12,20,40,100,200,600]
SNLEN_LIST = [2,5,20,100]
indlist = [0, 3, 13, 25]
# RETNOI_LIST = [0,1,2,3,4]
RETNOI_LIST = [0,1,2,3,4]
DIMENSION = 288
CATEGORY = False
TRANSFORMING_DATA = False    # whether to use generated xyc data or data transformed from forming data

PARAMETERS = system_params.SysParamSet()

# sfa1 = semantic.load_SFA(LOADPATH + "sfadef1train0.sfa")
delta_ret, delta_retmem, delta_seq, delta_retmemxyc, delta_retxyc, error_typesmem, error_distancesmem, error_types, error_distances = [], [], [], [], [], [], [], [], []

for ind, snlen in zip(indlist, SNLEN_LIST):
    delta_ret.append([])
    delta_retmem.append([])
    error_distancesmem.append([])
    error_typesmem.append([])
    error_distances.append([])
    error_types.append([])
    if TRANSFORMING_DATA:
        delta_retmemxyc.append([])
        delta_retxyc.append([])
    print("snlen = {}".format(snlen))
    npoints = 30000
    # snlen = 60
    nsnip = npoints//snlen
    retlen = 80
    nret = npoints//retlen
    # retnoi = 0.2

    # formdata = np.load(LOADPATH + "forming{}.npz".format(ind))
    # form_seq, form_cat, form_lat = formdata["forming_sequenceX"], formdata["forming_categories"], formdata["forming_latent"]
    # form_cat = np.array(form_cat)
    # form_lat = np.array(form_lat)
    #
    # formy = semantic.exec_SFA(sfa1, form_seq)
    # formy_w = streamlined.normalizer(formy, PARAMETERS.normalization)(formy)
    #
    # with open(LOADPATH+"learner{}.p".format(snlen), 'rb') as f:
    #     learner = pickle.load(f)
    #
    # predictionF = learner.predict(formy_w)
    # prediction_sequence = np.delete(predictionF, np.s_[2:4], 1)

    dx = 0.05
    step=5

    def _clamp(x, b): return np.clip(x, -b, b)
    def _randN(ds) : return np.random.normal(0, ds, DIMENSION)

    seq = []
    pats = []
    keys = []
    cat_current = 1
    for sni in range(nsnip):
        cat_current = (cat_current + 1) % 2
        xy = xy_new = 2*np.random.rand(DIMENSION)-1
        if CATEGORY:
            xy_new = np.append(xy, cat_current)
        if TRANSFORMING_DATA:
            xy_new = prediction_sequence[sni*snlen]
        seq.append(xy_new)
        pats.append(xy_new)
        for i in range(snlen-1):
            last = seq[-1]
            if CATEGORY:
                last = last[:-1]
            # xy = xy_new = _clamp(last + dx * _randN(step), 1)  # Adjust coefficient of step size for coverage
            xy = xy_new = 2 * np.random.rand(DIMENSION) - 1
            if CATEGORY:
                xy_new = np.append(xy, cat_current)
            if TRANSFORMING_DATA:
                xy_new = prediction_sequence[sni*snlen+i+1]
            seq.append(xy_new)
            keys.append(xy_new)
            if i < snlen-2:
                pats.append(xy_new)
        if DRAW:
            temp = np.array(seq[-snlen:])
            x = temp[:,0]
            y = temp[:,1]
            plt.plot(x,y,'bo-')
    pats = np.array(pats)
    keys = np.array(keys)

    if DRAW:
        plt.show()

    memory = episodic.EpisodicMemory(p_dim=DIMENSION+int(CATEGORY), retrieval_length=retlen, retrieval_noise=0)
    for sni in range(nsnip):
        memory.store_sequence(seq[sni*snlen:sni*snlen+snlen])

    # _ = memory.retrieve_sequence(0)
    # plt.hist(np.sqrt(memory.dmat[0]), bins=50)
    # plt.figure()
    # plt.hist(np.sqrt(scipy.spatial.distance.cdist(keys[0:1], pats, 'sqeuclidean')[0]), bins=50)
    # plt.show()

    for retnoi in RETNOI_LIST:
        ret = []
        retmem = []
        error_tmem = []
        error_dmem = []
        error_t = []
        error_d = []
        retmemlat = []
        retmemcat = []
        retlat = []
        retcat = []
        print("retnoi = {}".format(retnoi))
        for reti in range(nret):
            last_index = np.random.randint(len(keys))
            cue = keys[last_index]
            rets, retran, ertyp, erdis = memory.retrieve_sequence(cue, ret_noise=retnoi, return_indices=True)
            retmem.extend(rets)
            error_tmem.extend(ertyp)
            error_dmem.extend(erdis)
            # if TRANSFORMING_DATA:
            #     retmemcat.append(form_cat[retran])
            #     retmemlat.append(form_lat[retran])
            for i in range(retlen):
                if retnoi > 0:
            #         # noi = np.random.normal(0, retnoi, DIMENSION+int(CATEGORY))
                    noi = np.random.normal(0, retnoi, DIMENSION)
            #         noi = learner.predict(noi)
            #         noi = np.delete(noi, np.s_[2:4], 1)
                else:
            #         # noi = np.array([0.]*DIMENSION+int(CATEGORY))
                    noi = np.array([0.] * DIMENSION)
            #         noi = learner.predict(noi)
            #         noi = np.delete(noi, np.s_[2:4], 1)
                p = cue + noi
                dist_mat = np.sum((p-pats)**2, axis=1)
                mindex = np.argmin(dist_mat)
                ret.append(pats[mindex])
                if len(ret)-reti*retlen > 0:
                    retdis = np.linalg.norm(pats[mindex]-pats[last_index])
                    orgdis = np.linalg.norm(keys[last_index]-pats[last_index])
                    errdis=retdis-orgdis
                    error_d.append(errdis)
                    if errdis == 0:
                        error_t.append(0)
                    elif np.array_equal(keys[last_index], pats[(last_index+1)%len(pats)]):  #retrieval error was not forced by end of sequence
                        error_t.append(1)
                    else:
                        error_t.append(-1)
                last_index = mindex
                cue = keys[last_index]
            #     if TRANSFORMING_DATA:
            #         retlat.append(form_lat[mindex])
            #         retcat.append(form_cat[mindex])
        if TRANSFORMING_DATA:
            retmemlat = np.concatenate(retmemlat)
            retmemcat = np.concatenate(retmemcat)
            retlat = np.array(retlat)
            retcat = np.array(retcat)
            retmemxyc = np.append(np.delete(retmemlat, np.s_[2:4], 1), retmemcat[:, None], axis=1)
            retxyc = np.append(np.delete(retlat, np.s_[2:4], 1), retcat[:, None], axis=1)
            delta_retmemxyc[-1].append(tools.delta_diff(retmemxyc))
            delta_retxyc[-1].append(tools.delta_diff(retxyc))

        # delta_ret[-1].append(tools.delta_diff(ret))
        delta_retmem[-1].append(tools.delta_diff(retmem))
        delta_ret[-1].append(tools.delta_diff(ret))
        error_typesmem[-1].append(error_tmem)
        error_distancesmem[-1].append(error_dmem)
        error_types[-1].append(error_t)
        error_distances[-1].append(error_d)

    delta_seq.append(tools.delta_diff(seq))
delta_ret = np.array(delta_ret)
delta_retmem = np.array(delta_retmem)
error_typesmem = np.array(error_typesmem)
error_distancesmem = np.array(error_distancesmem)
error_types = np.array(error_types)
error_distances = np.array(error_distances)

# plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")
# for ni, nnn in enumerate(RETNOI_LIST):
#     plt.plot(SNLEN_LIST, np.mean(delta_ret[:, ni:ni+1, :], axis=2), ['y-', 'g-', 'm-', 'r-', 'c-'][ni], label="ret {}".format(nnn))
# plt.legend()
# plt.xscale('log')
#
# plt.figure()
# plt.plot(SNLEN_LIST, np.mean(delta_seq, axis=1), 'k-', label="seq")
# for ni, nnn in enumerate(RETNOI_LIST):
#     plt.plot(SNLEN_LIST, np.mean(delta_retmem[:, ni:ni+1, :], axis=2), ['y-', 'g-', 'm-', 'r-', 'c-'][ni], label="retmem {}".format(nnn))
# plt.legend()
# plt.xscale('log')
#
# plt.show()

np.save(PATH + "delta_seq.npy", np.array(delta_seq))
np.save(PATH + "delta_ret.npy", delta_ret)
np.save(PATH + "delta_retmem.npy", delta_retmem)
np.save(PATH + "error_distancesmem.npy", error_distancesmem)
np.save(PATH + "error_typesmem.npy", error_typesmem)
np.save(PATH + "error_distances.npy", error_distances)
np.save(PATH + "error_types.npy", error_types)
if TRANSFORMING_DATA:
    np.save(PATH + "delta_retxyc.npy", delta_retxyc)
    np.save(PATH + "delta_retmemxyc.npy", delta_retmemxyc)
# with open(PATH + "memory.p", 'wb') as f:
#     pickle.dump(memory,f)
