from core import sensory, system_params, input_params
import numpy as np
import sys
import pickle

PARAMETERS = system_params.SysParamSet()

sample_parmsT = dict(PARAMETERS.st1)
sample_parmsT['number_of_snippets'] = 1
sample_parmsT['movement_type'] = 'sample'
sample_parmsT['movement_params'] = dict()
sample_parmsT['object_code'] = input_params.make_object_code('T')
sample_parmsT['sequence'] = [0]

sample_parmsL = dict(sample_parmsT)
sample_parmsL['object_code'] = input_params.make_object_code('L')

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default)

sample_sequenceT, sample_categoriesT, sample_latentT = sensory_system.generate(fetch_indices = False, **sample_parmsT)
np.savez("../results/inp_sample/inp_sampleT.npz", sample_sequence=sample_sequenceT, sample_categories=sample_categoriesT, sample_latent=sample_latentT)

sample_sequenceL, sample_categoriesL, sample_latentL = sensory_system.generate(fetch_indices = False, **sample_parmsL)
np.savez("../results/inp_sample/inp_sampleL.npz", sample_sequence=sample_sequenceL, sample_categories=sample_categoriesL, sample_latent=sample_latentL)
