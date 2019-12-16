from core import semantic, sensory, system_params, streamlined, tools
import numpy as np
from matplotlib import pyplot as plt

# sfaparm1 = [
#         ('layer.linear', {
#             'bo':               30,
#             'rec_field_ch':     21,
#             'spacing':          9,
#             'in_channel_dim':   1,
#             'out_sfa_dim':     32
#         })]
#
# sfaparm2=[('single.linear', {
#             'dim_in': 128,
#             'dim_out': 48
#         })
#     ]

sfaparm1 = [
        ('layer.linear', {
            'bo':               30,
            'rec_field_ch':     14,
            'spacing':          8,
            'in_channel_dim':   1,
            'out_sfa_dim':     16
        })
    ]

sfaparm2 = [
        ('single.linear', {
            'dim_in': 144,
            'dim_out': 16
        })
    ]

PARAMETERS = system_params.SysParamSet()
# PARAMETERS.st1['movement_type'] = 'timed_border_stroll'
# PARAMETERS.st1['movement_params'] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=50, border_extent=2.3)
# PARAMETERS.st1['snippet_length'] = None
PARAMETERS.st1['movement_type'] = 'gaussian_walk'
PARAMETERS.st1['movement_params'] = dict(dx=0.05, dt=0.05, step=5)
PARAMETERS.st1['snippet_length'] = 100
PARAMETERS.st1['number_of_snippets'] = 500
PARAMETERS.st1['input_noise'] = 0

sfa1 = semantic.build_module(sfaparm1)
sfa2 = semantic.build_module(sfaparm2)
sensys = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
train_seq, train_cat, train_lat = sensys.generate(**PARAMETERS.st1)
form_seq, form_cat, form_lat = sensys.generate(**PARAMETERS.st1)
test_seq, test_cat, test_lat, test_ran = sensys.generate(**PARAMETERS.st1, fetch_indices=True)

cat_ind = [np.where(test_cat == 0)[0], np.where(test_cat == 1)[0]]

semantic.train_SFA(sfa1, train_seq)
yy = semantic.exec_SFA(sfa1, form_seq)
yw = streamlined.normalizer(yy, PARAMETERS.normalization)(yy)
yyt = semantic.exec_SFA(sfa1, test_seq)
ywt = streamlined.normalizer(yyt, PARAMETERS.normalization)(yyt)
zz = semantic.exec_SFA(sfa2, ywt)
zw = streamlined.normalizer(zz, PARAMETERS.normalization)(zz)
corr = tools.feature_latent_correlation(zw, test_lat, test_cat)
corrA = tools.feature_latent_correlation(zw[cat_ind[0]], list(np.array(test_lat)[cat_ind[0]]))
corrB = tools.feature_latent_correlation(zw[cat_ind[1]], list(np.array(test_lat)[cat_ind[1]]))

lis = ['corr', 'corrA', 'corrB']
f, ax = plt.subplots(len(lis), 1, squeeze=True)
for ki, key in enumerate(lis):
    ax[ki].matshow(np.abs(eval(key)), cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax[ki].set_title(key)
    for (ii, jj), z in np.ndenumerate(eval(key)):
        ax[ki].text(jj, ii, '{:.0f}'.format(z*100), ha='center', va='center', color="white")
plt.show()
