from core import semantic, sensory, system_params, input_params, streamlined, tools

import pickle
import numpy as np

sfaparm = [
        ('layer.square', {
            'bo':               30,
            'rec_field_ch':     15,
            'spacing':          3,
            'in_channel_dim':   1,
            'out_sfa_dim1':     48,
            'out_sfa_dim2':     32
        }),
        ('layer.square', {
            # <bo>=  (bo-rec_field_ch)/spacing+1 := 6
            'rec_field_ch':     4,
            'spacing':          2,
            'out_sfa_dim1':     8,
            'out_sfa_dim2':     32
        }),
        ('single.square', {
            'dim_in':      128,
            'dim_mid':     48,
            'dim_out':     32
        })
    ]

PARAMETERS = system_params.SysParamSet()

trainparm = dict(number_of_snippets=500, snippet_length=100, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TLKO'), sequence=[0,1,2,3], input_noise=0)

testparm = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TLKO'), sequence=[0,1,2,3], input_noise=0)

testparmT = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('T'), sequence=[0], input_noise=0)
testparmL = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('L'), sequence=[0], input_noise=0)
testparmK = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('K'), sequence=[0], input_noise=0)
testparmO = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('O'), sequence=[0], input_noise=0)

sensys = sensory.SensorySystem(PARAMETERS.input_params_default)

print("Generating testing data")
xx, cat, lat = sensys.generate(**testparm)

with open("/local/results/zack.p", 'rb') as f:
    res = pickle.load(f)
sfa = res['sfa']
print("Executing SFA")
yy = semantic.exec_SFA(sfa, xx)

cat_ind = [np.where(cat == 0)[0], np.where(cat == 1)[0], np.where(cat == 2)[0], np.where(cat == 3)[0]]

xx_w = streamlined.normalizer(xx, PARAMETERS.normalization)(xx)
xxT = xx_w[cat_ind[0]]
xxL = xx_w[cat_ind[1]]
xxK = xx_w[cat_ind[2]]
xxO = xx_w[cat_ind[3]]
yy_w = streamlined.normalizer(yy, PARAMETERS.normalization)(yy)
yyT = yy_w[cat_ind[0]]
yyL = yy_w[cat_ind[1]]
yyK = yy_w[cat_ind[2]]
yyO = yy_w[cat_ind[3]]

xd = tools.delta_diff(xx_w)
xTd = tools.delta_diff(xxT)
xLd = tools.delta_diff(xxL)
xKd = tools.delta_diff(xxK)
xOd = tools.delta_diff(xxO)
yd = tools.delta_diff(yy_w)
yTd = tools.delta_diff(yyT)
yLd = tools.delta_diff(yyL)
yKd = tools.delta_diff(yyK)
yOd = tools.delta_diff(yyO)

corr = tools.feature_latent_correlation(yy_w, lat, cat)
corrT = tools.feature_latent_correlation(yyT, list(np.array(lat)[cat_ind[0]]), list(np.array(cat)[cat_ind[0]]))
corrL = tools.feature_latent_correlation(yyL, list(np.array(lat)[cat_ind[1]]), list(np.array(cat)[cat_ind[1]]))
corrK = tools.feature_latent_correlation(yyK, list(np.array(lat)[cat_ind[2]]), list(np.array(cat)[cat_ind[2]]))
corrO = tools.feature_latent_correlation(yyO, list(np.array(lat)[cat_ind[3]]), list(np.array(cat)[cat_ind[3]]))

res = dict(xd=xd, xTd=xTd, xLd=xLd, xKd=xKd, xOd=xOd, yd=yd, yTd=yTd, yLd=yLd, yKd=yKd, yOd=yOd,
           corr=corr, corrT=corrT, corrL=corrL, corrK=corrK, corrO=corrO, sfa=sfa, sfaparm=sfaparm, trainparm=trainparm, testparm=testparm)

with open("/local/results/zack1.p", 'wb') as f:
    pickle.dump(res, f)
