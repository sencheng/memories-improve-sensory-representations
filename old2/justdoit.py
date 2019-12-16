from core import semantic, system_params, input_params, streamlined, result, semantic_params

from matplotlib import pyplot
import numpy as np

PARAMETERS = system_params.SysParamSet()

PARAMETERS.filepath_sfa1 = "../results/sfanew.sfa"

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = True, False

PARAMETERS.program_extent = 4
PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.sem_params2E = PARAMETERS.sem_params2S = semantic_params.make_layer_series(32,32,32,32,16,16)

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['weight_vector'] = [1] * 6
PARAMETERS.st2['memory']['completeness_weight'] = 3
PARAMETERS.st2['memory']['retrieval_noise'] = 0.1
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

res = streamlined.program(PARAMETERS)

DKEYS = ['sfa1', 'sfa2S', 'sfa2E', 'forming_Y', 'retrieved_Y', 'testingY', 'testingZ_S', 'testingZ_E']
DCOLORS = ['r','r','r','y','y','k','k','k']

f, ax = pyplot.subplots(1, 1, sharex=True, sharey=True)
for i, (k, c) in enumerate(zip(DKEYS, DCOLORS)):
    d = np.mean(res.d_values[k][:8])  # mean of first 8 d-values
    ax.bar([i + 0.1], [d], width=0.8, color=c)
    ax.text(i + 0.5, d, '{:.4f}'.format(d), ha='center', va='bottom', color="black", rotation=90)
tix = list(np.arange(len(DKEYS)) + 0.5)
ax.set_xticks(tix)
ax.set_xticklabels(DKEYS, rotation=70)
ax.set_yscale("log", nonposy='clip')