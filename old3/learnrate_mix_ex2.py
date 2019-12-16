from core import semantic, system_params, input_params, streamlined, sensory

import numpy as np
import sys
import pickle
import time
import os

def _randN(ds) : return np.random.normal(0, ds, 1)[0]

dim_id = int(sys.argv[1])
ei = int(sys.argv[2])
eps = float(sys.argv[3])
typ = sys.argv[4]
fram = int(sys.argv[5])
rot = sys.argv[6]

PATH = "../results/lromix_{}{}{}/".format(typ, fram, rot)

print("this is learnrate_mix_ex in folder " + PATH)

bsfa_parms1 = [
        ('single.linear', {
            'dim_in': 288,
            'dim_out': 16
        })
    ]

incsfa_parms1 = [
    ('inc.linear', {
        'dim_in': 288,
        'dim_out': 16
    })
]

bsfa_parms2 = None

incsfa_parms2 = None

PARAMETERS = system_params.SysParamSet()
PARAMETERS.st1.update(dict(number_of_snippets=100, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=0.1))

# poll for completion of pre script
while True:
    time.sleep(10)
    if os.path.isfile([PATH + "sfa1.p", PATH + "sfa2.p"][dim_id-1]):
        break
time.sleep(2)

sfa = semantic.load_SFA([PATH + "sfa1.p", PATH + "sfa2.p"][dim_id-1])

if ei == 0:
    bsfa = semantic.build_module([bsfa_parms1, bsfa_parms2][dim_id-1])
incsfa = semantic.build_module([incsfa_parms1, incsfa_parms2][dim_id-1], eps=eps)

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
sensysb = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

bparms = dict(PARAMETERS.st1)
bparms['number_of_snippets'] = 1200
_, _, training_latent, training_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
ran_arr = np.array(training_ranges)

copydict = dict(PARAMETERS.st1)
copydict['movement_type'] = 'copy_traj'
copydict['movement_params'] = dict(latent=training_latent, ranges=iter(training_ranges))
training_sequence, _, _ = sensory_system.generate(fetch_indices=False, **copydict)
copydict['movement_params'] = dict(latent=training_latent, ranges=iter(training_ranges))
copydict['object_code'] = input_params.make_object_code('T')
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
np.savez(PATH + "training_mix.npz", mixed_sequence=mixed_sequence, mixes=mixes, training_latent=training_latent)

_, _, blat, bran = sensysb.generate(fetch_indices=True, **bparms)

copydict = dict(bparms)
copydict['movement_type'] = 'copy_traj'
copydict['movement_params'] = dict(latent=blat, ranges=iter(bran))
bseq, _, _ = sensory_system.generate(fetch_indices=False, **copydict)
copydict['movement_params'] = dict(latent=blat, ranges=iter(bran))
copydict['object_code'] = input_params.make_object_code('T')
bseq2, _, _ = sensory_system.generate(fetch_indices=False, **copydict)
mixesb = []
mixed_sequenceb = []
for ir, r in enumerate(bran):
    mixesb.append([np.random.rand()])
    for i in range(len(r)-1):
        mix_value = mixesb[ir][i]+0.05*_randN(5)
        mixesb[ir].append(mix_value)
mixesb = np.array(mixesb).flatten()

for imv, mv in enumerate(mixesb):
    mixed_sequenceb.append(bseq[imv] * mv + bseq2[imv] * (1 - mv))
mixed_sequenceb = np.array(mixed_sequenceb)
np.savez(PATH + "training_mix_batch.npz", mixed_sequence=mixed_sequenceb, mixes=mixesb, training_latent=blat)

ran = np.arange(PARAMETERS.st1['number_of_snippets'])
for i in range(40):
    train_data = mixed_sequence
    seq = semantic.exec_SFA(sfa, train_data)
    seq_w = streamlined.normalizer(seq, PARAMETERS.normalization)(seq)
    if ei == 0 and i == 0:
        yb = semantic.exec_SFA(sfa, mixed_sequenceb)
        yb_w = streamlined.normalizer(yb, PARAMETERS.normalization)(yb)
        semantic.train_SFA(bsfa, yb_w)
        bsfa.save(PATH + "b{}.sfa".format(dim_id))
    semantic.train_SFA(incsfa, seq_w)
    incsfa.save(PATH + "inc{}_eps{}_{}.sfa".format(dim_id,ei,i))
    ran = np.random.permutation(PARAMETERS.st1['number_of_snippets'])
    inds = ran_arr[ran].flatten()
    train_data = train_data[inds]
