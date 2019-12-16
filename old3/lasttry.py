from core import sensory, system_params, semantic, streamlined, tools
import numpy as np
import pickle
import sklearn.linear_model
import scipy.stats
# from matplotlib import pyplot as plt

binwidth = 0.01
SAVEPATH = "../results/lasttrybig/"
LOADPATH = "../results/"

sfa2_parms = [
        ('inc.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

sfa2_normal = semantic.build_module(sfa2_parms, eps=0.0005)
sfa2_min = semantic.build_module(sfa2_parms, eps=0.0005)
sfa2_max = semantic.build_module(sfa2_parms, eps=0.0005)

sfa1 = semantic.load_SFA(LOADPATH+"reorder_o18a/sfadef1train0.sfa")
with open(LOADPATH+"reorder_o18a/res0.p", 'rb') as f:
    res = pickle.load(f)
whitener = res.whitener

PARAMETERS = system_params.SysParamSet()

sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

PARAMETERS.st1["movement_type"] = "gaussian_walk"
PARAMETERS.st1["movement_params"] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st1["number_of_snippets"] = 2500
PARAMETERS.st1["snippet_length"] = 50

print("Generating data...")
data, categories, latents = sensys.generate(**PARAMETERS.st1)
data_cat = np.array(categories)
data_lat = np.array(latents)

dataY = semantic.exec_SFA(sfa1, data)
dataYW = whitener(dataY)

print("Binning data...")
binedges = np.arange(-1, 1+binwidth, binwidth)  # how to divide x and y space in bins
cols = []                                       # first bin x in columns, after that bin y for each column
xsort_ind = np.argsort(data_lat[:,0])
xlat_sorted = data_lat[:,0][xsort_ind]
split_points = np.searchsorted(xlat_sorted, binedges)     # divide x values according to binedges
for i in range(len(split_points)-1):
    cols.append(xsort_ind[split_points[i]:split_points[i+1]])     # get element indices for each column

binned_inds = []    # resulting array
for ci, c in enumerate(cols):    # For every column this is the same as above
    binned_inds.append([])
    col_ysort_ind = np.argsort(data_lat[:,1][c])
    coly_sorted = data_lat[:,1][c][col_ysort_ind]
    split_points = np.searchsorted(coly_sorted, binedges)
    for i in range(len(split_points) - 1):
        binned_inds[ci].append(c[col_ysort_ind[split_points[i]:split_points[i + 1]]])
# binned_inds = np.array(binned_inds)

print("Generating training sequences...")
PARAMETERS.st1["number_of_snippets"] = 200
PARAMETERS.st1["snippet_length"] = 50
seq, cat, lat = sensys.generate(**PARAMETERS.st1)
cat_arr = np.array(cat)
lat_arr = np.array(lat)

seqY = semantic.exec_SFA(sfa1, seq)
seqYW = whitener(seqY)

print("Generating min and max sequences...")
ind = 0
seqind = PARAMETERS.st1["snippet_length"]
minseq = []
maxseq = []
mincat = []
maxcat = []
minlat = []
maxlat = []
while ind < len(cat_arr):
    if seqind == PARAMETERS.st1["snippet_length"]:   # start a new sequence
        minseq.append(seqYW[ind])        # the first element of a new sequence is always the first element of the corresponding original sequence
        maxseq.append(seqYW[ind])
        mincat.append(cat_arr[ind])
        maxcat.append(cat_arr[ind])
        minlat.append(lat_arr[ind])
        maxlat.append(lat_arr[ind])
        ind += 1                            # skip the element
        seqind = 1
    x = lat_arr[ind,0]
    y = lat_arr[ind,1]
    xind = np.searchsorted(binedges, x)-1
    yind = np.searchsorted(binedges, y)-1
    ind_bin = binned_inds[xind][yind]             # this is the currently relevant bin

    maxi = 0
    mini = 0
    maxdis = 0
    mindis = 99999999

    if len(ind_bin) > 1:         # only if we can select one max and one min candidate that are different from each other
        for b in ind_bin:
            dis_minseq = np.sum((minseq[-1] - dataYW[b]) ** 2)   # distance between last element of the sequence and the current element from the bin
            dis_maxseq = np.sum((maxseq[-1] - dataYW[b]) ** 2)
            if dis_maxseq > maxdis:
                maxi = b
                maxdis = dis_maxseq
            if dis_minseq < mindis:
                mini = b
                mindis = dis_minseq

        minseq.append(dataYW[mini])      # add the elements to the sequences
        maxseq.append(dataYW[maxi])
        mincat.append(data_cat[mini])
        maxcat.append(data_cat[maxi])
        minlat.append(data_lat[mini])
        maxlat.append(data_lat[maxi])
    else:
        minseq.append(seqYW[ind])
        maxseq.append(seqYW[ind])
        mincat.append(cat_arr[mini])
        maxcat.append(cat_arr[maxi])
        minlat.append(lat_arr[mini])
        maxlat.append(lat_arr[maxi])

    ind += 1
    seqind += 1

minseq = np.array(minseq)
maxseq = np.array(maxseq)
mincat = np.array(mincat)
maxcat = np.array(maxcat)
minlat = np.array(minlat)
maxlat = np.array(maxlat)

print("Training SFA2 normal...")
semantic.train_SFA(sfa2_normal, seqYW)
print("Training SFA2 min...")
semantic.train_SFA(sfa2_min, minseq)
print("Training SFA2 max...")
semantic.train_SFA(sfa2_max, maxseq)

seqZ = semantic.exec_SFA(sfa2_normal, seqYW)
minseqZ = semantic.exec_SFA(sfa2_min, minseq)
maxseqZ = semantic.exec_SFA(sfa2_max, maxseq)

print("Generating test sequences...")
PARAMETERS.st1["number_of_snippets"] = 200
PARAMETERS.st1["snippet_length"] = 50
testseq, testcat, testlat = sensys.generate(**PARAMETERS.st1)
testcat_arr = np.array(testcat)
testlat_arr = np.array(testlat)

print("Testing stuff...")
testseqY = semantic.exec_SFA(sfa1, testseq)
testseqYW = whitener(testseqY)

test_normal = semantic.exec_SFA(sfa2_normal, testseqYW)
test_min = semantic.exec_SFA(sfa2_min, testseqYW)
test_max = semantic.exec_SFA(sfa2_max, testseqYW)

test_normalW = streamlined.normalizer(test_normal, PARAMETERS.normalization)(test_normal)
test_minW = streamlined.normalizer(test_min, PARAMETERS.normalization)(test_min)
test_maxW = streamlined.normalizer(test_max, PARAMETERS.normalization)(test_max)

deltas_normal = tools.delta_diff(test_normalW)
deltas_min = tools.delta_diff(test_min)
deltas_max = tools.delta_diff(test_max)

dmean_normal = np.mean(np.sort(deltas_normal)[:3])
dmean_min = np.mean(np.sort(deltas_min)[:3])
dmean_max = np.mean(np.sort(deltas_min)[:3])

training_matrix = seqZ
target_matrix = np.append(lat_arr, cat_arr[:, None], axis=1)
learner_normal = sklearn.linear_model.LinearRegression()
learner_normal.fit(training_matrix, target_matrix)

training_matrix = minseqZ
target_matrix = np.append(minlat, mincat[:, None], axis=1)
learner_min = sklearn.linear_model.LinearRegression()
learner_min.fit(training_matrix, target_matrix)

training_matrix = maxseqZ
target_matrix = np.append(maxlat, maxcat[:, None], axis=1)
learner_max = sklearn.linear_model.LinearRegression()
learner_max.fit(training_matrix, target_matrix)

prediction = learner_normal.predict(test_normal)
_, _, rX_normal, _, _ = scipy.stats.linregress(testlat_arr[:, 0], prediction[:, 0])
_, _, rY_normal, _, _ = scipy.stats.linregress(testlat_arr[:, 1], prediction[:, 1])
_, _, rCAT_normal, _, _ = scipy.stats.linregress(testcat_arr, prediction[:, 4])

prediction = learner_min.predict(test_min)
_, _, rX_min, _, _ = scipy.stats.linregress(testlat_arr[:, 0], prediction[:, 0])
_, _, rY_min, _, _ = scipy.stats.linregress(testlat_arr[:, 1], prediction[:, 1])
_, _, rCAT_min, _, _ = scipy.stats.linregress(testcat_arr, prediction[:, 4])

prediction = learner_max.predict(test_max)
_, _, rX_max, _, _ = scipy.stats.linregress(testlat_arr[:, 0], prediction[:, 0])
_, _, rY_max, _, _ = scipy.stats.linregress(testlat_arr[:, 1], prediction[:, 1])
_, _, rCAT_max, _, _ = scipy.stats.linregress(testcat_arr, prediction[:, 4])

# f,ax = plt.subplots(1, 2, squeeze=True)
# barx = np.array([0.25, 0.75, 1.25, 2.25, 2.75, 3.25, 4.25, 4.75, 5.25])
# ax[0].bar(barx, [rX_min, rX_normal, rX_max, rY_min, rY_normal, rY_max, rCAT_min, rCAT_normal, rCAT_max], 0.5, tick_label=["Xmin", "Xnorm", "Xmax", "Ymin", "Ynorm", "Ymax", "IDmin", "IDnorm", "IDmax"])
# ax[1].bar([0.5, 1.5, 2.5], [dmean_min, dmean_normal, dmean_max], tick_label=["d min", "d norm", "d max"])
#
# plt.show()

np.savez(SAVEPATH+"data.npz", data=data, data_cat=data_cat, data_lat=data_lat)
np.savez(SAVEPATH+"train.npz", seq=seq, cat_arr=cat_arr, lat_arr=lat_arr)
np.savez(SAVEPATH+".train.npz", testseq=testseq, testcat=testcat, testlat=testlat)
sfa1.save(SAVEPATH+"sfa1.p")
sfa2_normal.save(SAVEPATH+"sfa2_normal.p")
sfa2_min.save(SAVEPATH+"sfa2_min.p")
sfa2_max.save(SAVEPATH+"sfa2_max.p")
with open(SAVEPATH+"learner_normal.p", 'wb') as f:
    pickle.dump(learner_normal, f)
with open(SAVEPATH+"learner_min.p", 'wb') as f:
    pickle.dump(learner_min, f)
with open(SAVEPATH+"learner_max.p", 'wb') as f:
    pickle.dump(learner_max, f)
np.savez(SAVEPATH+"deltas.npz", deltas_normal=deltas_normal, deltas_min=deltas_min, deltas_max=deltas_max)
np.savez(SAVEPATH+"rs.npz", rX_normal=rX_normal, rY_normal=rY_normal, rCAT_normal=rCAT_normal, rX_min=rX_min, rY_min=rY_min, rCAT_min=rCAT_min, rX_max=rX_max, rY_max=rY_max, rCAT_max=rCAT_max)
