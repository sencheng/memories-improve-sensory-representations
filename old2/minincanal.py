from core import semantic, sensory, system_params, input_params, tools, streamlined

import numpy as np
import sys
import pickle
from matplotlib import pyplot as plt

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1['number_of_snippets'] = 50
PARAMETERS.st1['input_noise'] = 0.2
PARAMETERS.st1['frame_shape'] = (12,12)
PARAMETERS.st1['object_code'] = input_params.make_object_code(['-'], sizes=22)
PARAMETERS.st1['sequence'] = [0]

sensory_system = pickle.load(open("/local/results/mininc/sensory.p", 'rb'))
ran = np.arange(PARAMETERS.st1['number_of_snippets'])
training_sequence, training_categories, training_latent = sensory_system.recall(numbers = ran, fetch_indices=False, **PARAMETERS.st1)

b1_sfa = semantic.load_SFA("/local/results/mininc/b1.sfa")
b2_sfa = semantic.load_SFA("/local/results/mininc/b2.sfa")
inc1_sfa = []
inc2_sfa = []
for i in range(20):
    inc1_sfa.append(semantic.load_SFA("/local/results/mininc/inc1_{}.sfa".format(i)))
    inc2_sfa.append(semantic.load_SFA("/local/results/mininc/inc2_{}.sfa".format(i)))

b1_y = semantic.exec_SFA(b1_sfa, training_sequence)
b1_w = streamlined.normalizer(b1_y, PARAMETERS.normalization)(b1_y)
b2_y = semantic.exec_SFA(b2_sfa, training_sequence)
b2_w = streamlined.normalizer(b2_y, PARAMETERS.normalization)(b2_y)
b1_ds = tools.delta_diff(b1_w)
print(len(b1_ds))
b1_d = np.mean(b1_ds)
b2_ds = tools.delta_diff(b2_w)
print(len(b2_ds))
b2_d = np.mean(b2_ds)

inc1_d = []
inc2_d = []
for i in range(20):
    inc1_y = semantic.exec_SFA(inc1_sfa[i], training_sequence)
    inc1_w = streamlined.normalizer(inc1_y, PARAMETERS.normalization)(inc1_y)
    inc1_ds = tools.delta_diff(inc1_w)
    inc1_d.append(np.mean(inc1_ds))
    inc2_y = semantic.exec_SFA(inc2_sfa[i], training_sequence)
    inc2_w = streamlined.normalizer(inc2_y, PARAMETERS.normalization)(inc2_y)
    inc2_ds = tools.delta_diff(inc2_w)
    inc2_d.append(np.mean(inc2_ds))

x = list(range(20))
lb1 = plt.plot(x,[b1_d]*20, label="b1")
linc1 = plt.plot(x,inc1_d, label="inc1")
lb2 = plt.plot(x,[b2_d]*20, label="b2")
linc2 = plt.plot(x,inc2_d, label="inc2")
plt.legend()
plt.show()