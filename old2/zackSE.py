from core import semantic, sensory, system_params, input_params, streamlined, tools

import pickle
import numpy as np
import time

sfaparm1 = [
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
            'out_sfa_dim2':     6
        })]

sfaparm2 = [
        ('inc.onesquare', {
            'dim_in':      24,
            'dim_out':     48
        })
    ]

PARAMETERS = system_params.SysParamSet()

trainparm = dict(number_of_snippets=250, snippet_length=200, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TMKC'), sequence=[0,1,2,3], input_noise=0)

testparm = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('TMKC'), sequence=[0,1,2,3], input_noise=0)

testparmT = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('T'), sequence=[0], input_noise=0.05)
testparmL = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('M'), sequence=[0], input_noise=0.05)
testparmK = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('K'), sequence=[0], input_noise=0.05)
testparmO = dict(number_of_snippets=50, snippet_length=50, movement_type='gaussian_walk', movement_params=dict(dx=0.05, dt=0.05, step=5),
                object_code=input_params.make_object_code('C'), sequence=[0], input_noise=0.05)

sensys = sensory.SensorySystem(PARAMETERS.input_params_default)

print("Generating training data")
x, _, _ = sensys.generate(**trainparm)
print("Generating forming data")
xf, _, _ = sensys.generate(**trainparm)
print("Generating testing data")
xx, cat, lat = sensys.generate(**testparm)

sfa1 = semantic.build_module(sfaparm1)
sfa2S = semantic.build_module(sfaparm2)
print("Training SFA1")
semantic.train_SFA(sfa1, x)
sfa1.save("../results/zackone2/zackoneSE_sfa1.sfa")
print("Executing SFA1")
yf = semantic.exec_SFA(sfa1, xf)
yf_w = streamlined.normalizer(yf, PARAMETERS.normalization)(yf)
print("Training SFA2")
semantic.train_SFA(sfa2S, yf_w)
sfa2S.save("../results/zackone2/zackoneSE_sfa2S.sfa")
time.sleep(5)
sfa2E = semantic.load_SFA("../results/zackone2/zackoneSE_sfa2S.sfa")
ran_array = np.reshape(np.arange(250 * 200), (250, 200))
for rep in range(39):
    print("SFA2E rep{}".format(rep+1))
    permran = np.random.permutation(len(ran_array))
    inds = ran_array[permran].flatten()
    new_yfw = yf_w[inds]
    semantic.train_SFA(sfa2E, new_yfw)
    sfa2E.save("../results/zackone2/zackoneSE_sfa2E_rep{}.sfa".format(rep + 1))
print("Executing SFA1")
yy = semantic.exec_SFA(sfa1, xx)
yy_w = streamlined.normalizer(yy, PARAMETERS.normalization)(yy)
print("Executing SFA2")
zzS = semantic.exec_SFA(sfa2S, yy_w)
zzE = semantic.exec_SFA(sfa2E, yy_w)
zzS_w = streamlined.normalizer(zzS, PARAMETERS.normalization)(zzS)
zzE_w = streamlined.normalizer(zzE, PARAMETERS.normalization)(zzE)


cat_ind = [np.where(cat == 0)[0], np.where(cat == 1)[0], np.where(cat == 2)[0], np.where(cat == 3)[0]]

xx_w = streamlined.normalizer(xx, PARAMETERS.normalization)(xx)
xxT = xx_w[cat_ind[0]]
xxM = xx_w[cat_ind[1]]
xxK = xx_w[cat_ind[2]]
xxC = xx_w[cat_ind[3]]

yyT = yy_w[cat_ind[0]]
yyM = yy_w[cat_ind[1]]
yyK = yy_w[cat_ind[2]]
yyC = yy_w[cat_ind[3]]

zzST = zzS_w[cat_ind[0]]
zzSM = zzS_w[cat_ind[1]]
zzSK = zzS_w[cat_ind[2]]
zzSC = zzS_w[cat_ind[3]]

zzET = zzE_w[cat_ind[0]]
zzEM = zzE_w[cat_ind[1]]
zzEK = zzE_w[cat_ind[2]]
zzEC = zzE_w[cat_ind[3]]

xd = tools.delta_diff(xx_w)
xTd = tools.delta_diff(xxT)
xMd = tools.delta_diff(xxM)
xKd = tools.delta_diff(xxK)
xCd = tools.delta_diff(xxC)
yd = tools.delta_diff(yy_w)
yTd = tools.delta_diff(yyT)
yMd = tools.delta_diff(yyM)
yKd = tools.delta_diff(yyK)
yCd = tools.delta_diff(yyC)

zSd = tools.delta_diff(zzS_w)
zSTd = tools.delta_diff(zzST)
zSMd = tools.delta_diff(zzSM)
zSKd = tools.delta_diff(zzSK)
zSCd = tools.delta_diff(zzSC)
zEd = tools.delta_diff(zzE_w)
zETd = tools.delta_diff(zzET)
zEMd = tools.delta_diff(zzEM)
zEKd = tools.delta_diff(zzEK)
zECd = tools.delta_diff(zzEC)

corrS = tools.feature_latent_correlation(zzS_w, lat, cat)
corrE = tools.feature_latent_correlation(zzE_w, lat, cat)
corrST = tools.feature_latent_correlation(zzST, list(np.array(lat)[cat_ind[0]]), list(np.array(cat)[cat_ind[0]]))
corrSM = tools.feature_latent_correlation(zzSM, list(np.array(lat)[cat_ind[1]]), list(np.array(cat)[cat_ind[1]]))
corrSK = tools.feature_latent_correlation(zzSK, list(np.array(lat)[cat_ind[2]]), list(np.array(cat)[cat_ind[2]]))
corrSC = tools.feature_latent_correlation(zzSC, list(np.array(lat)[cat_ind[3]]), list(np.array(cat)[cat_ind[3]]))
corrET = tools.feature_latent_correlation(zzET, list(np.array(lat)[cat_ind[0]]), list(np.array(cat)[cat_ind[0]]))
corrEM = tools.feature_latent_correlation(zzEM, list(np.array(lat)[cat_ind[1]]), list(np.array(cat)[cat_ind[1]]))
corrEK = tools.feature_latent_correlation(zzEK, list(np.array(lat)[cat_ind[2]]), list(np.array(cat)[cat_ind[2]]))
corrEC = tools.feature_latent_correlation(zzEC, list(np.array(lat)[cat_ind[3]]), list(np.array(cat)[cat_ind[3]]))

res = dict(xd=xd, xTd=xTd, xMd=xMd, xKd=xKd, xCd=xCd, yd=yd, yTd=yTd, yMd=yMd, yKd=yKd, yCd=yCd,
           zSd = zSd, zSTd = zSTd, zSMd = zSMd, zSKd = zSKd, zSCd = zSCd, zEd = zEd, zETd = zETd, zEMd = zEMd, zEKd = zEKd, zECd = zECd,
           corrS=corrS, corrST=corrST, corrSM=corrSM, corrSK=corrSK, corrSC=corrSC,
           corrE=corrE, corrET=corrET, corrEM=corrEM, corrEK=corrEK, corrEC=corrEC,
           sfa=sfa1, sfa2S=sfa2S, sfa2E=sfa2E, sfaparm1=sfaparm1, sfaparm2=sfaparm2, trainparm=trainparm, testparm=testparm,
           xxT=xx[cat_ind[0][:500]], xxM=xx[cat_ind[1][:500]], xxK=xx[cat_ind[2][:500]], xxC=xx[cat_ind[3][:500]])

with open("../results/zackone2/zackoneSE.p", 'wb') as f:
    pickle.dump(res, f)
