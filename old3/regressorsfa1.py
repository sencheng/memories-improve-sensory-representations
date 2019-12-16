from core import semantic, streamlined, system_params, result

import numpy as np
import sklearn.linear_model
import scipy.stats
import scipy.spatial.distance
from matplotlib import pyplot as plt
import pickle

PATH = "/local/results/reorder4a/"

noilist = [0.2, 1, 2, 3, 4]
lenlist = [2, 5, 20, 100]
indlist = [0, 3, 13, 25]

PARAMETERS = system_params.SysParamSet()

sfa1 = semantic.load_SFA(PATH + "sfadef1train0.sfa")

testdata = np.load(PATH + "testing0.npz")
test_seq, test_cat, test_lat = testdata["testing_sequenceX"], testdata["testing_categories"], testdata["testing_latent"]

f, ax = plt.subplots(4,1, sharex=True, squeeze=False)

for ind, leng in zip(indlist, lenlist):
    formdata = np.load(PATH + "forming{}.npz".format(ind))
    form_seq, form_cat, form_lat = formdata["forming_sequenceX"], formdata["forming_categories"], formdata["forming_latent"]

    training_matrix = semantic.exec_SFA(sfa1, form_seq)
    tr_w = streamlined.normalizer(training_matrix, PARAMETERS.normalization)(training_matrix)
    target_matrix = np.append(form_lat, form_cat[:, None], axis=1)
    learner = sklearn.linear_model.LinearRegression()
    learner.fit(tr_w, target_matrix)

    with open(PATH+"learner{}.p".format(leng), 'wb') as f:
        pickle.dump(learner, f)

    y = semantic.exec_SFA(sfa1, test_seq)
    y_w = streamlined.normalizer(y, PARAMETERS.normalization)(y)
    prediction = learner.predict(y_w)

    _, _, r_valueX, _, _ = scipy.stats.linregress(test_lat[:, 0], prediction[:, 0])
    _, _, r_valueY, _, _ = scipy.stats.linregress(test_lat[:, 1], prediction[:, 1])
    _, _, r_valueCAT, _, _ = scipy.stats.linregress(test_cat, prediction[:, 4])

    predictionF = learner.predict(tr_w)

    _, _, r_valueXF, _, _ = scipy.stats.linregress(form_lat[:, 0], predictionF[:, 0])
    _, _, r_valueYF, _, _ = scipy.stats.linregress(form_lat[:, 1], predictionF[:, 1])
    _, _, r_valueCATF, _, _ = scipy.stats.linregress(form_cat, predictionF[:, 4])

    # print("=== TESTING DATA ===")
    # print("r_valueX", r_valueX)
    # print("r_valueY", r_valueY)
    # print("r_valueCAT", r_valueCAT)
    # print("=== FORMING DATA ===")
    # print("r_valueXF", r_valueXF)
    # print("r_valueYF", r_valueYF)
    # print("r_valueCATF", r_valueCATF)

    prediction_sequence = np.delete(predictionF, np.s_[2:4], 1)
    xycat_sequence = np.append(form_lat[:,0:2], form_cat[:, None], axis=1)

    neighbors = []
    neighbors_real = []
    seqdis = []
    seqdis_real = []
    for i in range(30000//leng-1):
        for j in range(leng):
            if j != leng-1:
                neighbors.append(np.linalg.norm(prediction_sequence[leng*i+j]-prediction_sequence[leng*i+j+1]))
                neighbors_real.append(np.linalg.norm(xycat_sequence[leng * i + j] - xycat_sequence[leng * i + j + 1]))
            if i < 100:
                seqdis.append(np.delete(scipy.spatial.distance.cdist(prediction_sequence[leng*i+j:leng*i+j+1], prediction_sequence[leng*i:leng*(i+1)], 'euclidean').flatten(), j))
                seqdis_real.append(np.delete(scipy.spatial.distance.cdist(xycat_sequence[leng * i + j:leng * i + j + 1], xycat_sequence[leng * i:leng * (i + 1)], 'euclidean').flatten(), j))
    neighbors = np.array(neighbors)
    neighbors_real = np.array(neighbors_real)
    seqdis = np.array(seqdis).flatten()
    seqdis_real = np.array(seqdis_real).flatten()

    all100 = scipy.spatial.distance.cdist(prediction_sequence[0:100], prediction_sequence, 'euclidean').flatten()

    all100_real = scipy.spatial.distance.cdist(xycat_sequence[0:100], prediction_sequence, 'euclidean').flatten()

    ax[1, 0].set_yscale("log")
    ax[1, 0].hist(seqdis, bins=50, label=str(leng) ,alpha=0.4)
    ax[1, 0].set_title("sequence")
    ax[1, 0].legend(title="sequence length =")
    # ax[1, 1].set_yscale("log")
    # ax[1, 1].hist(seqdis_real, bins=50, label=str(leng), alpha=0.4)
    # ax[1, 1].set_title("sequence (real)")
    # ax[1,1].legend()
    ax[2, 0].hist(neighbors, bins=50, label=str(leng), alpha=0.4)
    ax[2, 0].set_title("neighbors")
    # ax[2, 1].hist(neighbors_real, bins=50, label=str(leng), alpha=0.4)
    # ax[2, 1].set_title("neighbors (real)")

# f.suptitle("sequence length = {}".format(leng))
ax[0,0].set_xlim([-0.5,6.5])
ax[0, 0].hist(all100, bins=50)
ax[0, 0].set_title("all")
# ax[0, 1].hist(all100_real, bins=50)
# ax[0, 1].set_title("all (real)")

n = []
nabs = []
nreal = []
nabsreal = []
for ni, noi in enumerate(noilist):
    for i in range(30000):
        n.append(np.random.normal(0, noi + 1e-20, len(y_w[0])))
        nreal.append(np.random.normal(0, noi + 1e-20, 3))
    prenoise = learner.predict(n)
    prenoise = np.delete(prenoise, np.s_[2:4], 1)
    nabs.append(np.linalg.norm(prenoise, axis=1))
    nabsreal.append(np.linalg.norm(nreal, axis=1))
    n = []
    nreal = []
    ax[3,0].hist(nabs[ni], bins=50, label=str(noi), alpha=0.4)
    # ax[3,1].hist(nabsreal[ni], bins=50, label=str(noi), alpha=0.4)
ax[3,0].set_title("noise")
# ax[3,1].set_title("noise (real)")
ax[3,0].legend(title="$\epsilon$=", loc=1)
# ax[3,1].legend()

# f, ax = plt.subplots(3,2)
# for c in range(3):
#     for r in range(2):
#         ax[c,r].hist(nabs[r*3+c], bins=50)
#         ax[c,r].set_title(noilist[r*3+c])

plt.show()
