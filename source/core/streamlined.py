# -*- coding: utf-8 -*-
"""
Useful for automatizing simulation runs.

"""

from . import semantic, sensory, episodic
from . import tools, result

import numpy as np
import random
import mdp
import sklearn.linear_model

def construct_SFA1(PARAMETERS, sensory_system=None):
    """
    Constructs an SFA with the given parameters: Generates training input, builds SFA and trains SFA.

    :param PARAMETERS: :py:class:`core.system_params.SysParamSet` instance. Need not to be complete but it is required to have
            three things: *sem_params1*, *input_params_default*, and *st1*. *st1* may be the empty dictionary.
    :param sensory_system: A SensorySystem object. If supplied, this object will be used
            to generate input.
    :returns: tuple of sfa, training_sequence, training_categories, training_latent.
            sfa is the trained SFA instance, training_sequence is the data that was used for training the SFA.
            training_categories and training_latent are the corresponding sequences of object identity labels and
            latent variables (x, y, cos(:math:`{\phi}`), sin(:math:`{\phi}`)), respectively.
            If PATAMETERS.same_input_for_all is True, the return tuple additionaly contains training_ranges, which is
            a list of ranges, one for each individual input episode in the generated data.
    """
    if not sensory_system: sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

    sfa1 = semantic.build_module(PARAMETERS.sem_params1)
    if PARAMETERS.same_input_for_all:
        training_sequence, training_categories, training_latent, training_ranges = sensory_system.generate(fetch_indices=True, **PARAMETERS.st1)
        semantic.train_SFA(sfa1, training_sequence)
        return sfa1, training_sequence, training_categories, training_latent, training_ranges

    training_sequence, training_categories, training_latent = sensory_system.generate(**PARAMETERS.st1)
    semantic.train_SFA(sfa1, training_sequence)

    return sfa1, training_sequence, training_categories, training_latent


class normalizer():
    def __init__(self, data, mode):
        """ Initialize a normalizer to perform whitening or scaling of the data.
            The transformation parameters are fixed on initialization such that on further data
            the exact same function is performed.

            For more information on whitening, look here:
            http://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening

            :param data: data to train the transformation parameters on.
            :param mode: normalization mode. String, either 'scale', 'whiten', 'whiten.ZCA', or 'none'.
            """

        self.mode = mode
        self.data_std = 0
        self.wn = None

        if self.mode == 'scale':
            self.data_std = np.std(data)

        elif self.mode.startswith('whiten.ZCA'):
            # original transmatrix.T =  wn.get_eigenvectors() * np.power(wn.d, -0.5)
            # new transform matrix.T =  wn.get_eigenvectors() * np.power(wn.d, -0.5) * wn.get_eigenvectors().T
            self.wn = mdp.nodes.WhiteningNode(svd=True)
            self.wn.train(data)
            self.wn.stop_training()

        elif self.mode.startswith('whiten'):
            self.wn = mdp.nodes.WhiteningNode(svd=True)
            self.wn.train(data)
            self.wn.stop_training()

    def __call__(self, X):
        """
        Execute the normalizer

        :param X: data to perform the transformation on.
        :return: transformed data
        """
        if self.mode == 'scale':
            return X / self.data_std

        elif self.mode.startswith('whiten.ZCA'):
            return (X-self.wn.avg).dot(self.wn.v.dot(self.wn.get_eigenvectors().T))

        elif self.mode.startswith('whiten'):
            return self.wn(X) # or equivalently, if we wanna treat everything equally:
            # return (X-self.wn.avg).dot(pcan.v)

        # Otherwise, if none of these modes fit, return the identity, that doesn't normalize
        return X


def program(PARAMETERS, sfa1=None, input=None, training_input=None, run_index=0):
    """
    Executes a simulation run.

    :param PARAMETERS: :py:class:`core.system_params.SysParamSet` instance.
    :param sfa1: None or a trained SFA module. If supplied, this will be used as SFA1 instead of training a new instance.
    :param input: None or input episodes and associated parameters for forming and testing data. If supplied,
                input must be of shape [(forming_sequenceX, forming_categories, forming_latent, forming_ranges),
                (testing_sequenceX, testing_categories, testing_latent)]
    :param training_input: None or array. Supplied training episodes can be used to make d_values of training data available in the returned Result object.
                They are not used for training SFA1, so an SFA1 instance that was trained with those episodes must be provided.
    :param run_index: Is only relevant if a grid of several programs is run to be able to identify individual runs.
    :returns: :py:class:`core.result.Result` object
    """
    # Give shorter names to the two super commonly used branch cases
    goal = PARAMETERS.program_extent
    S, E = 'S' in PARAMETERS.which, 'E' in PARAMETERS.which

    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)

    if sfa1 is None:
        if PARAMETERS.load_sfa1:
            print("Loading SFA1.. ", end='', flush=True)
            sfa1 = semantic.load_SFA(PARAMETERS.filepath_sfa1)
        else:
            print("Building SFA1.. ", end='', flush=True)
            if PARAMETERS.same_input_for_all:
                sfa1, training_sequence, training_categories, training_latent, training_ranges = construct_SFA1(PARAMETERS, sensory_system)
            else:
                sfa1, training_sequence, training_categories, training_latent = construct_SFA1(PARAMETERS, sensory_system)
        if PARAMETERS.save_sfa1:
            sfa1.save(PARAMETERS.filepath_sfa1)
    if goal >= 2 :
        sfa2S = semantic.build_module(PARAMETERS.sem_params2S, eps=PARAMETERS.st3['learnrate'])
        sfa2E = semantic.build_module(PARAMETERS.sem_params2E, eps=PARAMETERS.st3['learnrate'])
        if PARAMETERS.result_path != "":
            if PARAMETERS.st3['retr_repeats'] > 1:
                sfa2E_rep = semantic.build_module(PARAMETERS.sem_params2E, eps=PARAMETERS.st3['learnrate'])

        memory = episodic.EpisodicMemory(sfa1[-1].output_dim, **PARAMETERS.st2['memory'])

        print("Generating input.. ", end='', flush=True)
        whitening_sequenceX, _, _ = sensory_system.generate(**PARAMETERS.st4)

        if input is None:
            if PARAMETERS.same_input_for_all:
                forming_sequenceX, forming_categories, forming_latent, forming_ranges = training_sequence, training_categories, training_latent, training_ranges
                testing_sequenceX, testing_categories, testing_latent = training_sequence, training_categories, training_latent
                if PARAMETERS.st4b is not None:
                    testing_sequenceXb, testing_categoriesb, testing_latentb = training_sequence, training_categories, training_latent
            else:
                forming_sequenceX, forming_categories, forming_latent, forming_ranges = sensory_system.generate(
                    fetch_indices=True, **PARAMETERS.st2)
                testing_sequenceX, testing_categories, testing_latent = sensory_system.generate(**PARAMETERS.st4)
                if PARAMETERS.st4b is not None:
                    testing_sequenceXb, testing_categoriesb, testing_latentb = sensory_system.generate(**PARAMETERS.st4b)
        else:
            [forming_sequenceX, forming_categories, forming_latent, forming_ranges] = input[0]
            [testing_sequenceX, testing_categories, testing_latent] = input[1]
            if len(input) > 2:
                [testing_sequenceXb, testing_categoriesb, testing_latentb] = input[2]

        if training_input is not None:
            [training_sequence, training_categories, training_latent] = training_input

        forming_sequenceY = semantic.exec_SFA(sfa1, forming_sequenceX)

        whitening_sequenceX = forming_sequenceX
        whitening_sequenceY = semantic.exec_SFA(sfa1, whitening_sequenceX)
        whitener = normalizer(whitening_sequenceY, PARAMETERS.normalization)
        # forming_sequenceY = normalizer(forming_sequenceY, PARAMETERS.normalization)(forming_sequenceY)
        forming_sequenceY = whitener(forming_sequenceY)

    if goal >= 3:
        cat1 = np.array(forming_categories)
        lat1 = np.array(forming_latent)
        if E:
            lat = np.array(forming_latent)
            print("storing+scenarios... ",  end='', flush=True)
            for ran in forming_ranges:
                liran = list(ran)
                memory.store_sequence(forming_sequenceY[liran], categories=forming_categories[liran], latents=lat[liran])

            retrieved_sequence = []
            retrieved_indices = []
            retrieved_cat = []
            retrieved_lat = []
            if memory.return_err_values:
                error_types = []
                error_distances = []
            if PARAMETERS.generate_debug_presequences:
                perfect_presequence = []
                retrieved_presequence = []
                BLACK = np.ones(PARAMETERS.input_params_default['frame_shape'][0]*PARAMETERS.input_params_default['frame_shape'][1])*forming_sequenceX.max()

            if PARAMETERS.st3['cue_equally']:
                cat_count = len(np.unique(forming_categories))

            if PARAMETERS.st3['use_memory']:
                for j in range(PARAMETERS.st2['number_of_retrieval_trials']):
                    # cue is determined randomly. max value is determined by subtracting 2 times the retrieval length to make sure
                    # a perfect retrieval can take place. The factor 2 is determined arbitrarily just to make sure it also works
                    # with retrieval methods that may create longer sequences than retrieval length, e.g. including completeness!
                    cue_index = random.randint(0, len(forming_sequenceY) - 2*PARAMETERS.st2['memory']['retrieval_length'])
                    if PARAMETERS.st3['cue_equally']:
                        current_cat = j % cat_count
                        while forming_categories[cue_index] != current_cat:
                            cue_index = random.randint(0, len(forming_sequenceY) - 2 * PARAMETERS.st2['memory']['retrieval_length'])
                    if not memory.return_err_values:
                        ret_Y, ret_i = memory.retrieve_sequence(
                            forming_sequenceY[cue_index], forming_categories[cue_index], return_indices=True)
                    else:
                        ret_Y, ret_i, err_types, err_dist  = memory.retrieve_sequence(
                            forming_sequenceY[cue_index], forming_categories[cue_index], return_indices=True)

                    #print( cue_index, np.shape(ret_Y), np.shape(ret_i), np.shape(forming_sequenceX))

                    retrieved_sequence.append(ret_Y)
                    retrieved_indices.append(ret_i)
                    if memory.return_err_values:
                        error_distances.append(err_dist)
                        error_types.append(err_types)
                    if PARAMETERS.generate_debug_presequences:
                        perfect_presequence.append(forming_sequenceX[range(cue_index, cue_index+len(ret_i))])
                        retrieved_presequence.append(forming_sequenceX[ret_i])
                        perfect_presequence.append(np.array([BLACK]*5))
                        retrieved_presequence.append(np.array([BLACK]*5))
                    retrieved_cat.append(cat1[ret_i])
                    retrieved_lat.append(lat1[ret_i])

                retrieved_sequence = np.concatenate(retrieved_sequence, axis=0)
                retrieved_indices = np.concatenate(retrieved_indices, axis=0)
                if memory.return_err_values:
                    error_distances = np.concatenate(error_distances, axis=0)
                    error_types = np.concatenate(error_types, axis=0)
                if PARAMETERS.generate_debug_presequences:
                    retrieved_presequence = np.vstack(retrieved_presequence)
                    perfect_presequence = np.vstack(perfect_presequence)
                ret_cat = np.concatenate(retrieved_cat)
                ret_lat = np.concatenate(retrieved_lat, axis=0)

                if PARAMETERS.result_path != "":
                    if PARAMETERS.st3['retr_repeats'] > 1:
                        if type(sfa2E_rep) == semantic.incflow:
                            if PARAMETERS.st2['sfa2_noise'] > 0:
                                retrieved_sequenceN = retrieved_sequence + np.random.normal(0, PARAMETERS.st2['sfa2_noise'], retrieved_sequence.shape)  # Add noise to sfa2 training data
                            else:
                                retrieved_sequenceN = retrieved_sequence
                            semantic.train_SFA(sfa2E_rep, retrieved_sequenceN)
                            sfa2E_rep.save(PARAMETERS.result_path + 'sfa2E_res{}_retr{}.sfa'.format(run_index, 0))
                            for i in range(1, PARAMETERS.st3['retr_repeats']):
                                retrieved_sequence_rep = []
                                for j in range(PARAMETERS.st2['number_of_retrieval_trials']):
                                    cue_index = random.randint(0, len(forming_sequenceY) - 2 * PARAMETERS.st2['memory']['retrieval_length'])
                                    if PARAMETERS.st3['cue_equally']:
                                        current_cat = j % cat_count
                                        while forming_categories[cue_index] != current_cat:
                                            cue_index = random.randint(0, len(forming_sequenceY) - 2 * PARAMETERS.st2['memory']['retrieval_length'])
                                    ret_Y, ret_i = memory.retrieve_sequence(
                                        forming_sequenceY[cue_index], forming_categories[cue_index], return_indices=True)
                                    retrieved_sequence_rep.append(ret_Y)
                                retrieved_sequence_rep = np.concatenate(retrieved_sequence_rep, axis=0)
                                if PARAMETERS.st2['sfa2_noise'] > 0:
                                    retrieved_sequence_rep += np.random.normal(0, PARAMETERS.st2['sfa2_noise'], retrieved_sequence_rep.shape)  # Add noise to sfa2 training data
                                semantic.train_SFA(sfa2E_rep, retrieved_sequence_rep)
                                sfa2E_rep.save(PARAMETERS.result_path + 'sfa2E_res{}_retr{}.sfa'.format(run_index, i))
            else:
                retrieved_sequence = forming_sequenceY

        print("training SFA2s... ", end='', flush=True)
        if S:
            if PARAMETERS.st2['sfa2_noise']>0:
                forming_sequenceY += np.random.normal(0, PARAMETERS.st2['sfa2_noise'], forming_sequenceY.shape)  # Add noise to sfa2 training data
            semantic.train_SFA(sfa2S, forming_sequenceY)
            if PARAMETERS.result_path != "":
                if PARAMETERS.st3['inc_repeats_S'] > 1:
                    if type(sfa2S) == semantic.incflow:
                        sfa2S.save(PARAMETERS.result_path+'sfa2S_res{}_repeat{}.sfa'.format(run_index, 0))
                        ran_array = np.array(forming_ranges)
                        for i in range(1, PARAMETERS.st3['inc_repeats_S']):
                            permran = np.random.permutation(len(ran_array))
                            inds = ran_array[permran].flatten()
                            new_formingY = forming_sequenceY[inds]
                            semantic.train_SFA(sfa2S, new_formingY)
                            sfa2S.save(PARAMETERS.result_path + 'sfa2S_res{}_repeat{}.sfa'.format(run_index, i))
                    else:
                        sfa2S.save(PARAMETERS.result_path + 'sfa2S_res{}_batch.sfa'.format(run_index, 0))

            target_matrixS = np.append(lat1, cat1[:, None], axis=1)
            training_matrixS = semantic.exec_SFA(sfa2S, forming_sequenceY)
            learnerS = sklearn.linear_model.LinearRegression()
            learnerS.fit(training_matrixS, target_matrixS)

            print("(S) ", end='', flush=True)
        if E:
            if PARAMETERS.st2['sfa2_noise']>0:
                retrieved_sequence += np.random.normal(0, PARAMETERS.st2['sfa2_noise'], retrieved_sequence.shape)  # Add noise to sfa2 training data
            semantic.train_SFA(sfa2E, retrieved_sequence)
            if PARAMETERS.result_path != "":
                if PARAMETERS.st3['inc_repeats_E'] > 1:
                    if type(sfa2E) == semantic.incflow:
                        sfa2E.save(PARAMETERS.result_path+'sfa2E_res{}_repeat{}.sfa'.format(run_index, 0))
                        ran_array = np.reshape(np.arange(PARAMETERS.st2['number_of_retrieval_trials']*PARAMETERS.st2['memory']['retrieval_length']),
                                               (PARAMETERS.st2['number_of_retrieval_trials'], PARAMETERS.st2['memory']['retrieval_length']))
                        for i in range(1, PARAMETERS.st3['inc_repeats_E']):
                            permran = np.random.permutation(len(ran_array))
                            inds = ran_array[permran].flatten()
                            new_retrieved = retrieved_sequence[inds]
                            semantic.train_SFA(sfa2E, new_retrieved)
                            sfa2E.save(PARAMETERS.result_path + 'sfa2E_res{}_repeat{}.sfa'.format(run_index, i))
                    else:
                        sfa2E.save(PARAMETERS.result_path + 'sfa2E_res{}_batch.sfa'.format(run_index, 0))

            target_matrixE = np.append(ret_lat, ret_cat[:, None], axis=1)
            training_matrixE = semantic.exec_SFA(sfa2E, retrieved_sequence)
            learnerE = sklearn.linear_model.LinearRegression()
            learnerE.fit(training_matrixE, target_matrixE)

            print("(E)... ", end='', flush=True)

    # ============================ STAGE 4: TESTING===============================================
    if goal >= 4:
        print("testing...", end='', flush=True)

        # run SFA chain for both cases
        testing_sequenceY = semantic.exec_SFA(sfa1, testing_sequenceX)
        # testing_sequenceY = normalizer(testing_sequenceY, PARAMETERS.normalization)(testing_sequenceY)
        testing_sequenceY = whitener(testing_sequenceY)
        if PARAMETERS.st4['sfa2_noise'] > 0:
            testing_sequenceY += np.random.normal(0, PARAMETERS.st2['sfa2_noise'], testing_sequenceY.shape)  # Add noise before sfa2

        # Normalize with the normalization mode. We throw away the normalizer and run directly on data (for optimal comparison of d-values).
        if S:
            testing_sequenceZ_S = semantic.exec_SFA(sfa2S, testing_sequenceY)
            testing_sequenceZ_S = normalizer(testing_sequenceZ_S, PARAMETERS.normalization)(testing_sequenceZ_S)
        if E:
            testing_sequenceZ_E = semantic.exec_SFA(sfa2E, testing_sequenceY)
            testing_sequenceZ_E = normalizer(testing_sequenceZ_E, PARAMETERS.normalization)(testing_sequenceZ_E)

        # storage
        if PARAMETERS.st4['do_memtest']:
            if S: memtestS = episodic.EpisodicMemory(sfa2S[-1].output_dim, **PARAMETERS.st4['memtest'])
            if E: memtestE = episodic.EpisodicMemory(sfa2E[-1].output_dim, **PARAMETERS.st4['memtest'])
            if S: memtestS.store_sequence(testing_sequenceZ_S)
            if E: memtestE.store_sequence(testing_sequenceZ_E)

            # retrieval
            retrieval_length = PARAMETERS.st4['memtest']['retrieval_length']
            if S: retr_simple, jumps_simple, indices_simple, org_simple = [], [], [], []
            if E: retr_episodic, jumps_episodic, indices_episodic, org_episodic =  [], [], [], []

            for i in range(PARAMETERS.st4['number_of_retrievals']):
                cue_index = random.randint(0, len(testing_sequenceZ_S) - retrieval_length)

                if S:
                    retr_simple_s, jumps_simple_s, indices_simple_s = memtestS.retrieve_sequence(
                            cue_index, return_jumps=True, return_indices=True)
                    org_simple_s = testing_sequenceZ_S[cue_index:cue_index + retrieval_length]

                    retr_simple.append(retr_simple_s)
                    jumps_simple.append(jumps_simple_s)
                    indices_simple.append(indices_simple_s)
                    org_simple.append(org_simple_s)

                if E:
                    retr_episodic_s, jumps_episodic_s, indices_episodic_s = memtestE.retrieve_sequence(
                            cue_index, return_jumps=True, return_indices=True)
                    org_episodic_s = testing_sequenceZ_E[cue_index:cue_index + retrieval_length]

                    retr_episodic.append(retr_episodic_s)
                    jumps_episodic.append(jumps_episodic_s)
                    indices_episodic.append(indices_episodic_s)
                    org_episodic.append(org_episodic_s)

        # REPEAT FOR testingB IF NEEDED
        if PARAMETERS.st4b is not None:
            # run SFA chain for both cases
            testing_sequenceYb = semantic.exec_SFA(sfa1, testing_sequenceXb)
            # testing_sequenceYb = normalizer(testing_sequenceYb, PARAMETERS.normalization)(testing_sequenceYb)
            if PARAMETERS.st4b['sfa2_noise'] > 0:
                testing_sequenceYb += np.random.normal(0, PARAMETERS.st2['sfa2_noise'], testing_sequenceYb.shape)  # Add noise before sfa2

            # Normalize with the normalization mode. We throw away the normalizer and run directly on data.
            if S:
                testing_sequenceZ_Sb = semantic.exec_SFA(sfa2S, testing_sequenceYb)
                testing_sequenceZ_Sb = normalizer(testing_sequenceZ_Sb, PARAMETERS.normalization)(testing_sequenceZ_Sb)
            if E:
                testing_sequenceZ_Eb = semantic.exec_SFA(sfa2E, testing_sequenceYb)
                testing_sequenceZ_Eb = normalizer(testing_sequenceZ_Eb, PARAMETERS.normalization)(testing_sequenceZ_Eb)

            # storage
            if PARAMETERS.st4b['do_memtest']:
                if S: memtestSb = episodic.EpisodicMemory(sfa2S[-1].output_dim, **PARAMETERS.st4b['memtest'])
                if E: memtestEb = episodic.EpisodicMemory(sfa2E[-1].output_dim, **PARAMETERS.st4b['memtest'])
                if S: memtestSb.store_sequence(testing_sequenceZ_Sb)
                if E: memtestEb.store_sequence(testing_sequenceZ_Eb)

                # retrieval
                retrieval_lengthb = PARAMETERS.st4b['memtest']['retrieval_length']
                if S: retr_simpleb, jumps_simpleb, indices_simpleb, org_simpleb = [], [], [], []
                if E: retr_episodicb, jumps_episodicb, indices_episodicb, org_episodicb = [], [], [], []

                for i in range(PARAMETERS.st4b['number_of_retrievals']):
                    cue_index = random.randint(0, len(testing_sequenceZ_Sb) - retrieval_lengthb)

                    if S:
                        retr_simple_s, jumps_simple_s, indices_simple_s = memtestS.retrieve_sequence(
                            cue_index, return_jumps=True, return_indices=True)
                        org_simple_s = testing_sequenceZ_Sb[cue_index:cue_index + retrieval_lengthb]

                        retr_simpleb.append(retr_simple_s)
                        jumps_simpleb.append(jumps_simple_s)
                        indices_simpleb.append(indices_simple_s)
                        org_simpleb.append(org_simple_s)

                    if E:
                        retr_episodic_s, jumps_episodic_s, indices_episodic_s = memtestE.retrieve_sequence(
                            cue_index, return_jumps=True, return_indices=True)
                        org_episodic_s = testing_sequenceZ_Eb[cue_index:cue_index + retrieval_lengthb]

                        retr_episodicb.append(retr_episodic_s)
                        jumps_episodicb.append(jumps_episodic_s)
                        indices_episodicb.append(indices_episodic_s)
                        org_episodicb.append(org_episodic_s)

    result_obj = result.Result(PARAMETERS, locals(), normalizer)

    print("FINISHED")

    return result_obj

if __name__ == '__main__':
    import system_params, result

    curvars = globals()
    if not 'PARAMS' in curvars:
        print("No parameters found. Creating default parameter set.")
        PARAMS = system_params.SysParamSet()
    else:
        print("PARAMS variable found. Using this for program parameters.")

    if not 'sfa' in curvars:
        print("No SFA found. Training one from parameter data.")
        sfa = None
    else:
        print("\"sfa\" found in local variables ... using  this to run program.")

    rslt = program(PARAMS, sfa)
