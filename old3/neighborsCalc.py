from core import streamlined, semantic, system_params
import numpy as np
import scipy.spatial.distance
from matplotlib import pyplot as plt
import matplotlib
import pickle


FORMPATH = "/local/results/reorder_o18a/"

# PATH_PRE = "/media/goerlrwh/Extreme Festplatte/results/"
PATH_PRE = "/local/results/"
PATH = PATH_PRE + "reorder_o18a/"

# Whitening settings
# of the data
WHITENER = False
NORM = False

LOAD = True

ZOOM = 500
NFEAT = 288

normcolor = '0.4'
normcolor_line = '0.4'
zoomcolor = 'k'
zoomcolor_line = 'k'

# if LOAD:
#     difmean = np.load(PATH+"difmean.npy")
#     difmeanout = np.load(PATH+"difmeanout.npy")
# else:
params = system_params.SysParamSet()

forming_data = np.load(FORMPATH+"forming0.npz")
if WHITENER:
    with open(PATH+"res0_whitener.p", "rb") as f:
        whitener = pickle.load(f)
formx = forming_data["forming_sequenceX"]

sfa1 = semantic.load_SFA(PATH+"sfadef1train0.sfa")
formy = semantic.exec_SFA(sfa1, formx)
if WHITENER:
    formyw = whitener(formy)
else:
    formyw = streamlined.normalizer(formy, params.normalization)(formy)

if WHITENER or NORM:
    keys = formyw[1::2][:,:NFEAT]
    pats = formyw[::2][:,:NFEAT]
else:
    keys = formy[1::2][:,:NFEAT]
    pats = formy[::2][:,:NFEAT]

# We look at all pattern-key pairs A,B
direction = []  # is positive if A->B (the vector B minus A) is pointing away from the distribution center M, and negative otherwise
direction_sides = []  # the same measure, but does only include those A,B-pairs that are on the same side of the distribution, relative to the center M.
                        # exludes all pairs for which (B-M)dot(A-M)<0 holds.

away = 0
towards = 0
away_invar = 0
towards_invar = 0

M = np.mean(pats,axis=0)  # mean of the data (middle of the data points distribution)
# for all pattern-key pairs A,B
for pi in range(len(pats)):
    A = pats[pi]
    B = keys[pi]
    val = np.dot(B-M, B-A)
    val_rev = np.dot(A-M, A-B)
    if val < 0:
        towards += 1
        if val_rev < 0:
            towards_invar += 1
    else:
        if val_rev > 0:
            away_invar += 1
        away += 1
    direction.append(val)
    tmp = np.dot(B-M, A-M)
    if not tmp < 0:
        direction_sides.append(val)

dir_mean = np.mean(direction)
dir_sides_mean = np.mean(direction_sides)

print("NFEAT", NFEAT)

print("average pat length", np.mean(np.linalg.norm(pats, axis=1)))

print("len(direction)", len(direction))
print("len(direction_sides)", len(direction_sides))

print("dir_mean", dir_mean)
print("dir_sides_mean", dir_sides_mean)

print("towards", towards)
print("towards_invar", towards_invar)
print("away", away)
print("away_invar", away_invar)
