from core import semantic, system_params, input_params, streamlined, sensory, tools

import numpy as np
import sklearn.linear_model
import scipy.stats
from matplotlib import pyplot as plt

PATH = "/local/results/testmix/"

def _randN(ds) : return np.random.normal(0, ds, 1)[0]

sfa1 = semantic.load_SFA(PATH+"sfa1.sfa")
bsfa = semantic.load_SFA(PATH+"bsfa.sfa")
incsfaS = semantic.load_SFA(PATH+"incsfaS.sfa")
incsfaE = semantic.load_SFA(PATH+"incsfaE.sfa")

PARAMETERS = system_params.SysParamSet()
PARAMETERS.st1.update(dict(number_of_snippets=1000, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=0.1))

try:
    dataTest = np.load(PATH + "testing_mix.npz")
    mixed_sequence = dataTest["mixed_sequence"]
    mixes = dataTest["mixes"]
    training_latent = dataTest["training_latent"]
except:

    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
    sensysb = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

    print("Generating test latents...")
    _, _, training_latent, training_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
    ran_arr = np.array(training_ranges)
    copydict = dict(PARAMETERS.st1)
    copydict['movement_type'] = 'copy_traj'
    copydict['movement_params'] = dict(latent=training_latent, ranges=iter(training_ranges))
    print("Generating test L data...")
    training_sequence, _, _ = sensory_system.generate(fetch_indices=False, **copydict)

    copydict['movement_params'] = dict(latent=training_latent, ranges=iter(training_ranges))
    copydict['object_code'] = input_params.make_object_code('T')
    print("Generating test T data...")
    training_sequence2, _, _ = sensory_system.generate(fetch_indices=False, **copydict)
    mixes = []
    mixed_sequence = []
    for ir, r in enumerate(training_ranges):
        mixes.append([np.random.rand()])
        for i in range(len(r)-1):
            mix_value = mixes[ir][i]+0.05*_randN(5)
            mixes[ir].append(mix_value)
    mixes = np.array(mixes).flatten()

    for imv, mv in enumerate(mixes):
        mixed_sequence.append(training_sequence[imv] * mv + training_sequence2[imv] * (1 - mv))
    mixed_sequence = np.array(mixed_sequence)
    np.savez(PATH + "testing_mix.npz", mixed_sequence=mixed_sequence, mixes=mixes, training_latent=training_latent)

dataS = np.load(PATH+"training_mix.npz")
seqS = dataS["mixed_sequence"]
catS = dataS["mixes"]
latS = dataS["training_latent"]
dataE = np.load(PATH+"training_mix_batch.npz")
seqE = dataE["mixed_sequence"]
catE = dataE["mixes"]
latE = dataE["training_latent"]

print("Running SFAs...")
yE = semantic.exec_SFA(sfa1, seqE)
yEw = streamlined.normalizer(yE, PARAMETERS.normalization)(yE)
yS = semantic.exec_SFA(sfa1, seqS)
ySw = streamlined.normalizer(yS, PARAMETERS.normalization)(yS)
zB = semantic.exec_SFA(bsfa, yEw)
zE = semantic.exec_SFA(incsfaE, yEw)
zS = semantic.exec_SFA(incsfaS, ySw)

yTest = semantic.exec_SFA(sfa1, mixed_sequence)
yTestw = streamlined.normalizer(yTest, PARAMETERS.normalization)(yTest)
zTestB = semantic.exec_SFA(bsfa, yTestw)
zTestS = semantic.exec_SFA(incsfaS, yTestw)
zTestE = semantic.exec_SFA(incsfaE, yTestw)

print("Training regressors...")
learnerb = sklearn.linear_model.LinearRegression()
learnerS = sklearn.linear_model.LinearRegression()
learnerE = sklearn.linear_model.LinearRegression()
targetE = np.append(latE, catE[:,None], axis=1)
targetS = np.append(latS, catS[:,None], axis=1)
learnerb.fit(zB, targetE)
learnerE.fit(zE, targetE)
learnerS.fit(zS, targetS)

predictionb = learnerb.predict(zTestB)
predictionS = learnerb.predict(zTestS)
predictionE = learnerb.predict(zTestE)

_, _, r_valueXb, _, _ = scipy.stats.linregress(training_latent[:, 0], predictionb[:, 0])
_, _, r_valueYb, _, _ = scipy.stats.linregress(training_latent[:, 1], predictionb[:, 1])
_, _, r_valueCb, _, _ = scipy.stats.linregress(mixes, predictionb[:, 4])

_, _, r_valueXS, _, _ = scipy.stats.linregress(training_latent[:, 0], predictionS[:, 0])
_, _, r_valueYS, _, _ = scipy.stats.linregress(training_latent[:, 1], predictionS[:, 1])
_, _, r_valueCS, _, _ = scipy.stats.linregress(mixes, predictionS[:, 4])

_, _, r_valueXE, _, _ = scipy.stats.linregress(training_latent[:, 0], predictionE[:, 0])
_, _, r_valueYE, _, _ = scipy.stats.linregress(training_latent[:, 1], predictionE[:, 1])
_, _, r_valueCE, _, _ = scipy.stats.linregress(mixes, predictionE[:, 4])

print("============================")
print("{:4}{:10}{:10}{:10}".format("", "batch", "incS", "incE"))
print("{:<4}{:10.3f}{:10.3f}{:10.3f}".format("x", r_valueXb, r_valueXS, r_valueXE))
print("{:<4}{:10.3f}{:10.3f}{:10.3f}".format("y", r_valueYb, r_valueYS, r_valueYE))
print("{:<4}{:10.3f}{:10.3f}{:10.3f}".format("c", r_valueCb, r_valueCS, r_valueCE))
