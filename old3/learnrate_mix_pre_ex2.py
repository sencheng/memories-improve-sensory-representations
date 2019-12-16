from core import semantic, sensory, system_params, input_params

import pickle, os, sys, numpy as np

def _randN(ds) : return np.random.normal(0, ds, 1)[0]

typ = sys.argv[1]
fram = sys.argv[2]
rot = sys.argv[3]

PATH = "../results/lromix_{}{}{}/".format(typ, fram, rot)

if not os.path.isdir(PATH):
    os.system("mkdir " + PATH)

print("this is learnrate_mix_pre in folder " + PATH)

if typ == 'o14':
    print("set SFA parms to o14")
    sfa1_parms = [
            ('layer.linear', {
                'bo':               30,
                'rec_field_ch':     14,
                'spacing':          8,
                'in_channel_dim':   1,
                'out_sfa_dim':     32
            })
        ]

else:
    print("set SFA parms to o18")
    sfa1_parms = [
        ('layer.linear', {
            'bo': 30,
            'rec_field_ch': 18,
            'spacing': 6,
            'in_channel_dim': 1,
            'out_sfa_dim': 32
        })
    ]

sfa2parms = None

if fram == 50:
    nsnip = 50
    snlen = 50
else:
    nsnip = 600
    snlen = 100

if rot == 't':
    dehteh = 0.05
else:
    dehteh = 0.0

# if typ == 'j':
#     noi = 0.2
# else:
#     noi = 0
noi = 0.1

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1.update(dict(number_of_snippets=2*nsnip, snippet_length=snlen, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=dehteh, step=5),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=noi))
# PARAMETERS.st1['number_of_snippets'] = 50
# PARAMETERS.st1['input_noise'] = 0.2

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
print("Generating input")
_, _, training_latent, training_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
pickle.dump(sensory_system, open(PATH + "sensory.p", 'wb'))
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
np.savez(PATH + "pre_training_mix.npz", mixed_sequence=mixed_sequence, mixes=mixes, training_latent=training_latent)

sfa1 = semantic.build_module(sfa1_parms)
# sfa2 = semantic.build_module(sfa2_parms)

semantic.train_SFA(sfa1, training_sequence)
# semantic.train_SFA(sfa2, training_sequence)

sfa1.save(PATH + "sfa1.p")
# sfa2.save(PATH + "sfa2.p")

with open(PATH + "st1.p", 'wb') as f:
    pickle.dump(PARAMETERS.st1, f)
