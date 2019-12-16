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
            'out_sfa_dim1':     48,
            'out_sfa_dim2':     32
        }),
        ('single.square', {
            'dim_in':      128,
            'dim_mid':     48,
            'dim_out':     48
        })
    ]

PARAMETERS = system_params.SysParamSet()

trainparmS = dict(number_of_snippets=250, snippet_length=200, movement_type='uniform', movement_params=dict(),
                object_code=input_params.make_object_code('TMKC'), sequence=[0,1,2,3], input_noise=0)
trainparmE = dict(number_of_snippets=250, snippet_length=200, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TMKC'), sequence=[0,1,2,3], input_noise=0)

testparm = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TMKC'), sequence=[0,1,2,3], input_noise=0)

testparmT = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('T'), sequence=[0], input_noise=0)
testparmL = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('M'), sequence=[0], input_noise=0)
testparmK = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('K'), sequence=[0], input_noise=0)
testparmO = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('C'), sequence=[0], input_noise=0)

sensys = sensory.SensorySystem(PARAMETERS.input_params_default)

print("Generating training data")
xS, _, _ = sensys.generate(**trainparmS)
xE, _, _ = sensys.generate(**trainparmE)
print("Generating testing data")
xx, cat, lat = sensys.generate(**testparm)

sfaS = semantic.build_module(sfaparm)
sfaE = semantic.build_module(sfaparm)
print("Training SFA")
semantic.train_SFA(sfaS, xS)
semantic.train_SFA(sfaE, xE)
print("Executing SFA")
yyS = semantic.exec_SFA(sfaS, xx)
yyE = semantic.exec_SFA(sfaE, xx)

cat_ind = [np.where(cat == 0)[0], np.where(cat == 1)[0], np.where(cat == 2)[0], np.where(cat == 3)[0]]

xx_w = streamlined.normalizer(xx, PARAMETERS.normalization)(xx)
xxT = xx_w[cat_ind[0]]
xxM = xx_w[cat_ind[1]]
xxK = xx_w[cat_ind[2]]
xxC = xx_w[cat_ind[3]]
yyS_w = streamlined.normalizer(yyS, PARAMETERS.normalization)(yyS)
yyST = yyS_w[cat_ind[0]]
yySM = yyS_w[cat_ind[1]]
yySK = yyS_w[cat_ind[2]]
yySC = yyS_w[cat_ind[3]]
yyE_w = streamlined.normalizer(yyE, PARAMETERS.normalization)(yyE)
yyET = yyE_w[cat_ind[0]]
yyEM = yyE_w[cat_ind[1]]
yyEK = yyE_w[cat_ind[2]]
yyEC = yyE_w[cat_ind[3]]

xd = tools.delta_diff(xx_w)
xTd = tools.delta_diff(xxT)
xMd = tools.delta_diff(xxM)
xKd = tools.delta_diff(xxK)
xCd = tools.delta_diff(xxC)
ySd = tools.delta_diff(yyS_w)
ySTd = tools.delta_diff(yyST)
ySMd = tools.delta_diff(yySM)
ySKd = tools.delta_diff(yySK)
ySCd = tools.delta_diff(yySC)
yEd = tools.delta_diff(yyE_w)
yETd = tools.delta_diff(yyET)
yEMd = tools.delta_diff(yyEM)
yEKd = tools.delta_diff(yyEK)
yECd = tools.delta_diff(yyEC)

corrS = tools.feature_latent_correlation(yyS_w, lat, cat)
corrST = tools.feature_latent_correlation(yyST, list(np.array(lat)[cat_ind[0]]), list(np.array(cat)[cat_ind[0]]))
corrSM = tools.feature_latent_correlation(yySM, list(np.array(lat)[cat_ind[1]]), list(np.array(cat)[cat_ind[1]]))
corrSK = tools.feature_latent_correlation(yySK, list(np.array(lat)[cat_ind[2]]), list(np.array(cat)[cat_ind[2]]))
corrSC = tools.feature_latent_correlation(yySC, list(np.array(lat)[cat_ind[3]]), list(np.array(cat)[cat_ind[3]]))
corrE = tools.feature_latent_correlation(yyE_w, lat, cat)
corrET = tools.feature_latent_correlation(yyET, list(np.array(lat)[cat_ind[0]]), list(np.array(cat)[cat_ind[0]]))
corrEM = tools.feature_latent_correlation(yyEM, list(np.array(lat)[cat_ind[1]]), list(np.array(cat)[cat_ind[1]]))
corrEK = tools.feature_latent_correlation(yyEK, list(np.array(lat)[cat_ind[2]]), list(np.array(cat)[cat_ind[2]]))
corrEC = tools.feature_latent_correlation(yyEC, list(np.array(lat)[cat_ind[3]]), list(np.array(cat)[cat_ind[3]]))

res = dict(xd=xd, xTd=xTd, xMd=xMd, xKd=xKd, xCd=xCd, ySd=ySd, ySTd=ySTd, ySMd=ySMd, ySKd=ySKd, ySCd=ySCd, yEd=yEd, yETd=yETd, yEMd=yEMd, yEKd=yEKd, yECd=yECd,
           corrS=corrS, corrST=corrST, corrSM=corrSM, corrSK=corrSK, corrSC=corrSC, corrE=corrE, corrET=corrET, corrEM=corrEM, corrEK=corrEK, corrEC=corrEC,
           sfaS=sfaS, sfaE=sfaE, sfaparm=sfaparm, trainparmS=trainparmS, trainparmE=trainparmE, testparm=testparm,
           xxT=xx[cat_ind[0][:500]], xxM=xx[cat_ind[1][:500]], xxK=xx[cat_ind[2][:500]], xxC=xx[cat_ind[3][:500]])

with open("../results/zackbSE1u.p", 'wb') as f:
    pickle.dump(res, f)
