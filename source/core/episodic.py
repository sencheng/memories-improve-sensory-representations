"""
Contains the :py:class:`EpisodicMemory` class that implements the sequence storage algorithm.

"""

import numpy as np
import mdp
import random
from . import tools
import scipy.spatial.distance
import pdb

class EpisodicMemory():
    """
    .. class:: EpisodicMemory

    """
    def __init__(self, p_dim, retrieval_length, retrieval_noise, category_weight=0,
                 weight_vector=1, depress_params=None, smoothing_percentile = 100, optimization=True, use_latents=False, return_err_values=False):
        """Initialize episodic memory with given parameters

        :param p_dim: dimensions of the patterns to store
        :param ret_len: length of retrieved sequences. Must be given, but can be changed when calling retrieval function.
        :param ret_noise: standard deviation of Gaussian retrieval noise. Must be given, but can be changed when calling retrieval function.
        :param category_weight: Memory can store an array of object identity labels. During retrieval, object identity is used as an additional pattern vector entry.
                                category weight determines how strongly that entry is weighted during retrieval.
                                A high category weight makes object transitions during a retrieved sequence more improbable.
                                If 0, no identity labels are stored.
        :param weight_vector: Vector determining how pattern vector entries are weighted during retrieval. If scalar, all elements are weighted equally.
                              If weight_vector has less elements than a pattern, remaining elements are weighted with 0. E.g. [1, 1, 1, 1] takes only the first four elements into account
                              for distance calculations during retrieval.
        :param depress_params: If not None, dictionary containing the keys 'cost' and 'recovery_time_constant', corresponding to :math:`{\\alpha}` and b, respectively
                               in the depression equation. 'Cost' is added to the depression value for a pattern after it is retrieved and 'recovery_time_constant' determines by what factor
                               the values decline in each retrieval step. Depression is not applied if depress_params is None.
        :param smoothing_percentile: If the distance between the retrieved pattern and the associated key is larger than the given percentile of the distribution of
                                     all pattern-key distances, the retrieved pattern itself is used as a cue instead of the associated key. Introduces additional smoothing by episodic memory.
        :param optimization: If True, only those neighbors of the key are considered for retrieval that are mathematically possible to retrieve,
                             depending on the level of retrieval noise and depression. For a high number of patterns in memory, this option can save up to 80% runtime.
        :param use_latents: If True, latent variables are used for distance calculations during retrieval instead of the patterns themselves.
                            Latents have to be provided to the store_sequence function (parameter 'latents')
        :param return_err_values: If True, retrieve_sequence additionally returns arrays error_types, error_distances,
                                  containing information about whether a retrieval error occured or the end of
                                  a stored sequence was reached (1 or -1 in error_types, respectively) and the retrieval offset for each retrieved element.
        """
        self.category_weight = category_weight
        self.pattern_dim = p_dim
        self.retrieval_length = retrieval_length
        self.retrieval_noise_level = retrieval_noise
        self.smoothing_percentile = smoothing_percentile
        self.optimization = optimization
        self.use_latents = use_latents
        self.return_err_values = return_err_values

        self.opti_values = []

        # if list is too short, zeros are appended, if too long, it is just cut off at the end. If single value,
        # that is used for all features. By putting [1,1,1,1], only the first 4 features are considered for retrieval.
        if hasattr(weight_vector, '__iter__'):
            if len(weight_vector) >= self.pattern_dim:
                self.retrieval_weights = np.resize(np.array(weight_vector), self.pattern_dim)
                self.use_helper_array = (0 in self.retrieval_weights)
                # helper array will be a copy of memory array, but elements corresponding to 0-weights will be removed. Speeds up calculation.
            else:
                self.retrieval_weights = np.array(weight_vector)
                self.use_helper_array = True
        else:
            self.retrieval_weights = np.array([weight_vector] * self.pattern_dim)
            self.use_helper_array = False
        # for memory and category a list and an ndarray is stored because appending of sequences is way faster for lists,
        # but vector-calculations are only efficient on ndarrays.
        self.memory_list = []  # actual episodic memory as list of patterns for storage
        if self.use_latents:
            self.latents_list = []
        self.memory_length = 0  # cumulative length of episodes in memory
        self.lexicon_of_indices = []  # the mighty look-up list of indices - to translate internal indices to total sequence indices. Because we remove the last element of each sequence.
        if self.use_helper_array:
            self.helper_mask = self.retrieval_weights > 0
            self.helper_dim = len(np.where(True == self.helper_mask)[0])
            self.do_retrieval_weighting = not (1 == len(np.unique(self.retrieval_weights[
                                                                      self.helper_mask])))  # if retrieval weights are binary, which is most likely, weighting during retrieval is not neccessary
            self.helper_list = []
        else:  # if helper array is not used, it is set to reference to memory array - so memory array is used for retrieval
            self.helper_dim = self.pattern_dim
            self.do_retrieval_weighting = not (1 == len(np.unique(self.retrieval_weights)))
            self.helper_list = self.memory_list
        self.retrieval_weights = self.retrieval_weights[np.where(
            self.retrieval_weights != 0)]  # remove 0 elements because they will not be considered in helper array

        self.depression = depress_params is not None
        if self.depression:
            self.depression_arr = np.empty((0, 1))
            self.depression_use_cost = depress_params['cost']
            #self.depression_afun = eval(depress_params.get('activation_function','lambda X: X'))  # CARE: afun is not used in current implementation
            self.depression_recovery = float(depress_params['recovery_time_constant'])

        self.changed = False  # keeps track whether memory_arr matches memory_list or if a conversion is necessary
        self.use_categories = self.category_weight > 0
        if self.use_categories:
            self.category_list = []  # list of category labels, one label corresponds to one frame

    def store_sequence(self, seq, categories=None, latents=None):
        """Store sequence in episodic memory. Create tuple of (pattern, key_to_next_pattern) for each element in sequence.
        
        :param seq: sequence to store (ndarray)
        :param categories: None or object identity label sequence to store (ndarray), one label for each pattern in seq.
                           Has to be given if memory was set to use object identity labels.
        :param latents: Noneo or latent variable sequence if network is set to do distance calculations based on latent variables instead of patterns.
        """
        if not len(seq[0]) == self.pattern_dim:
            raise Exception("pattern dimension mismatch")
        if self.use_categories:
            if categories is None:
                raise Exception("Memory is set so store categories, but None given.")
            self.category_list.append(categories)

        self.memory_list.append(seq)  # add to list of arrays
        if self.use_latents:
            self.latents_list.append(latents)
        if self.use_helper_array:
            self.helper_list.append(np.array([pat[self.helper_mask] for pat in
                                              seq]))  # pat as output op mdp flow should be numpy array, so list indexing is possible
        self.lexicon_of_indices.extend([x + self.memory_length for x in range(len(
            seq) - 1)])  # beginning at the index for the case without omitting last element, append increasing index sequence.
        # appended index sequence is 1 shorter than original sequence, because later in retrieval, last element of each sequence is omitted.
        self.memory_length = self.memory_length + len(seq)
        self.changed = True

    def retrieve_sequence(self, cue_or_idx, cue_category=None, ret_len=None, ret_noise=None, return_jumps=False,
                          return_indices=False, return_opti_values=False):
        """Retrieve sequence from episodic memory given a cue.

        :param cue_or_idx: pattern as ndarray with correct number of dimensions, or index of cue to use from memory.
                           Note that if a pattern is given, this pattern actually has to be taken from the stored episodes.
        :param cue_category: None or object identity label of given cue. Has to be given if memory is set to use identity labels.
        :param ret_len: None or length of retrieved sequence. If None, keep former setting.
        :param ret_noise: None or standard deviation of Gaussian retrieval noise. If None, keep former setting.
        :param return_jumps: If True, function returns tuple (sequence, jumps, [indices], [opti_values]).
                             jumps contains the distance (in memory indices) of the retrieved pattern from the cue for each sequence element. 0 for no error.
                             Always overridden with False if EpisodicMemory instance was initialized with return_err_values set to True.
        :param return_indices: If True, function returns tuple (sequence, [jumps], indices, [opti_values]).
                               indices contains the index within memory for each retrieved pattern of the sequence.
                               Always overridden with True if EpisodicMemory instance was initialized with return_err_values set to True.
        :param return_opti_values: If true, function returns tuple (sequence, [jumps], [indices], opti_values).
                                   opti_values contains list [search_radius, index_range] for each retrieved element.
                                   It contains information about how many neighbors of the cue are considered for retrieval by the optimization algorithm. For debugging purposes.
                                   Always overridden with False if EpisodicMemory instance was initialized with return_err_values set to True.
        :returns: retrieved_sequence, [jumps], [indices] (depending on parameters)
        """
        # setup parameters and make memory arrays
        if ret_len is not None:
            self.retrieval_length = ret_len  # if given, overwrite retrieval parameters
        if ret_noise is not None:
            self.retrieval_noise_level = ret_noise
        if self.changed:  # if list was changed,
            self.changed = False
            if len(self.memory_list) == 1:
                self.memory_list[0] = np.append(self.memory_list[0], [self.memory_list[0][-1]],
                                                axis=0)  # if only one sequence was stored, we want to detect if we reach the end of the sequence, so we need to include last element
                if self.use_latents:
                    self.latents_list[0]= np.append(self.latents_list[0], [self.latents_list[0][-1]], axis=0)
                if self.use_helper_array:
                    self.helper_list[0] = np.append(self.helper_list[0], [self.helper_list[0][-1]], axis=0)
                self.lexicon_of_indices.append(self.lexicon_of_indices[
                                                   -1])  # then we also need one more element in the mighty look-up list of indices

            self.pat_arr = np.concatenate([seq[:-1] for seq in
                                           self.memory_list])  # memory of full patterns, last element is not included. It appears only as key paired with the second-to-last pattern
            if self.use_latents:
                self.lat_arr = np.concatenate([seq[:-1] for seq in
                                           self.latents_list])
                self.lat_arr_keys = np.concatenate([seq[1:] for seq in self.latents_list])
            if self.use_helper_array:
                self.helper_arr = np.concatenate(
                    [seq[:-1] for seq in self.helper_list])  # memory of incomplete patterns, according to weight-vector
                self.helper_arr_keys = np.concatenate([seq[1:] for seq in
                                                       self.helper_list])  # incomplete keys. The first element of each sequence is omitted since there is no previous element to it
            else:
                self.helper_arr = self.pat_arr
                self.helper_arr_keys = np.concatenate([seq[1:] for seq in self.memory_list])  # full memory keys
            if self.use_categories:
                if cue_category is None:
                    raise Exception("Memory is set to store categories, but no category for cue given.")
                if len(self.memory_list) == 1:
                    self.category_list[0] = np.append(self.category_list[0], [self.category_list[0][-1]],
                                                      axis=0)  # only one sequence, repeat last element, as for sequences above
                self.category_arr = self.category_weight * np.concatenate([cat_list[:-1] for cat_list in
                                                                           self.category_list])  # as it is done for sequences, remove last element from cat array
                self.category_arr_keys = self.category_weight * np.concatenate(
                    [cat_list[1:] for cat_list in self.category_list])

            if self.depression:
                self.depression_arr = np.zeros(self.helper_arr.shape[0])

            if self.optimization:

                numframes = len(self.helper_arr)
                self.dmat = []
                self.dsort = []
                pair_dists = []

                if self.do_retrieval_weighting:
                    if self.use_categories:
                        key_arr = np.append(self.retrieval_weights * self.helper_arr_keys, self.category_arr_keys[:, None], axis=1)
                        pat_arr = np.append(self.retrieval_weights * self.helper_arr, self.category_arr[:, None], axis=1)  # SQUARED distances of all keys to all patterns with category
                    else:
                        key_arr = self.retrieval_weights * self.helper_arr_keys
                        pat_arr = self.retrieval_weights * self.helper_arr  # and without cat
                else:
                    if self.use_categories:
                        key_arr = np.append(self.helper_arr_keys, self.category_arr_keys[:, None], axis=1)
                        pat_arr = np.append(self.helper_arr, self.category_arr[:, None], axis=1)  # SQUARED distances of all keys to all patterns with category
                    else:
                        key_arr = self.helper_arr_keys
                        pat_arr = self.helper_arr     # and without cat

                # calculating dmat here (pairwise distances for all patterns. Doing it line by line because otherwise we get memory error for a large number of frames.
                for fram in range(numframes):
                    self.dmat.append(scipy.spatial.distance.cdist(key_arr[fram:fram + 1], pat_arr, 'sqeuclidean')[0])
                    self.dsort.append(np.argsort(self.dmat[fram])) # sorted indices for each key-patterns distance vector
                    pair_dists.append(self.dmat[fram][fram])

                self.smoothing_threshold = np.percentile(pair_dists, self.smoothing_percentile)

        # preparation of retrieval
        retrieved_sequence = []  # initialize retrieved sequence

        if self.return_err_values:
            error_types = []
            error_distances = []

        if return_jumps:
            jumps = []  # initialize array to hold jump size
        if return_indices:
            indices_ret = []  # initialize array to hold index for each returned sequence element

        if hasattr(cue_or_idx,
                   '__iter__'):  # test if cue (array) or index of cue (integer) is given and get cue accordingly
            cue = cue_or_idx[:self.pattern_dim]
            if self.use_helper_array:
                cue = cue[self.helper_mask]
            last_index = -1  # initialize last index parameter, saving the index resulting from the last for iteration
        else:
            cue = self.helper_arr[cue_or_idx]
            last_index = cue_or_idx - 1  # cue_ind is determined later based on last_index+1

        retrieval_counter = 0

        def getCueIndex(cue, ind, cat_cue=None, usekeys=False):
            aoi = self.helper_arr if not usekeys else self.helper_arr_keys  # array of interest
            if cat_cue is not None:
                coi = self.category_arr if not usekeys else self.category_arr_keys  # category_array of interest
                if ind < len(self.helper_arr) and np.array_equal(aoi[ind, :], cue) and coi[ind] == cat_cue:
                    return ind
            elif ind < len(self.helper_arr) and np.array_equal(aoi[ind, :], cue):
                return ind
            # cue_ind_cand = np.unique(np.nonzero(aoi == cue)[0])   #this was wrong. a match in just one vector element was enough here. this happens rarely for float but it does.
            if cat_cue is not None:
                cue_ind_cand = \
                np.nonzero((np.equal(np.append(aoi, coi[:, None], axis=1), np.append(cue, cat_cue))).all(axis=1))[0]
            else:
                cue_ind_cand = np.nonzero((np.equal(aoi, cue)).all(axis=1))[0]
            if len(cue_ind_cand) == 0:
                return -1  # return -1 if cue was not found in memory. This may happen when the end of a subsequence is reached, because the last element is omitted
                # returning -1 instead of 0 because this way the jump can be set to 0 manually. Otherwise this may result in really unrealistic jump sizes.
                # -1 is unproblematic here, because cue_ind is never used as an actual index to access data.
            cue_ind_ind = np.argmin(
                abs(cue_ind_cand - ind))  # take cue index with smallest deviation from strict sequential retrieval
            cue_ind = cue_ind_cand[cue_ind_ind]
            return cue_ind

        if self.use_categories:
            cue_category = self.category_weight if cue_category else 0

        # ACUTAL RETRIEVAL BEGINS HERE
        while True:
            if retrieval_counter == 0 and self.depression:
                # add depression for cue to avoid loop (and depression-driven retrieval errors type 1 for low noise)
                if last_index != -1:
                    cue_ind = last_index
                else:
                    if self.use_categories:
                        cue_ind = getCueIndex(cue, 0, cue_category, usekeys=False)
                    else:
                        cue_ind = getCueIndex(cue, 0, usekeys=False)
                self.depression_arr[cue_ind] += self.depression_use_cost * np.exp(-1 / self.depression_recovery)

            if last_index == -1:  # last_index is -1 when retrieval_counter is 0
                if self.use_categories:
                    last_index = getCueIndex(cue, last_index + 1, cue_category, usekeys=True)
                else:
                    last_index = getCueIndex(cue, last_index + 1, usekeys=True)
                    # last_index is now still -1 if a first sequence element was given as cue

            n = np.random.normal(0, self.retrieval_noise_level + 1e-20, self.helper_dim)
            p = cue + n

            if self.use_categories:
                #nc = self.category_weight * random.gauss(0, self.retrieval_noise_level + 1e-20)
                nc = random.gauss(0, self.retrieval_noise_level + 1e-20)
                p_cat = cue_category + nc

            if self.optimization:
                if not last_index == -1:  # last index can be -1 only in the first iteration, if a cue was chosen that does not exist as key (first sequence element)
                    d0 = self.dmat[last_index][self.dsort[last_index][0]]  # most probably 0, but not when key at last_index is one of the omitted last sequence elements
                    nn = np.append(n, nc) if self.use_categories else n
                    search_radius = 4* (np.sqrt(d0) + np.sqrt(np.sum(nn ** 2)) )**2

                    if self.use_latents:
                        cue_from_keys = True  # default
                else:
                    # when retrieval_counter is 0 and a first sequence element was given as cue
                    # the cue is probably similar to its associated key, so we look for the index
                    if self.use_categories:
                        last_index = getCueIndex(cue, -1,
                                                 cue_category)  # this time, last_index is actually not the last index
                    else:
                        last_index = getCueIndex(cue, -1)  # same here
                    cue_corr_key = self.helper_arr_keys[last_index]
                    # last_index ist used later twice:
                    # 1. selecting row from dsort, here we just need it to select the row corresponding to cue_corr_key
                    # 2. if several candidates with minimum distance are found, select the closest, no problem because cue anyway was not in memory as key
                    dis = np.sum((cue_corr_key - cue) ** 2)
                    d0 = self.dmat[last_index][self.dsort[last_index][0]]
                    nn = np.append(n, nc) if self.use_categories else n
                    search_radius = 4*(np.sqrt(d0) + np.sqrt(np.sum(nn ** 2)) + np.sqrt(dis) )**2

                    if self.use_latents:
                        cue_from_keys = False   # exceptional cases

            if not self.use_latents:
                if self.optimization:
                    if self.depression:
                        search_radius += 4 * np.abs(np.max(self.depression_arr))

                    def findsmart(dma, dso, d):
                        """given a number d and an unsorted distance vector dma and an accordingly argsorted index-vector dso
                        finding the index of dso with the largest value in dma that is smaller as or equal to d."""
                        l = len(dso)
                        if l == 1:
                            return 0
                        ii = l // 2
                        if dma[dso[ii]] > d:
                            return findsmart(dma, dso[:ii], d)
                        else:
                            return ii + findsmart(dma, dso[ii:], d)

                    index_range = 1 + findsmart(self.dmat[last_index], self.dsort[last_index],
                                                search_radius)  # function returns index, so add one to use as range boundary

                    if return_opti_values:
                        self.opti_values.append([search_radius, index_range])

                    index_array = self.dsort[last_index][:index_range]
                else:
                    index_array = np.arange(len(self.helper_arr))    #if optimization is turned off (debug purposes) take entire array
                part_arr = self.helper_arr[
                    index_array]  # take only part of the memory into account, according to index_array

                def make_dist_mat():
                    di = part_arr - p
                    if self.do_retrieval_weighting:
                        mul = self.retrieval_weights * di
                    else:
                        mul = di
                    di2 = mul ** 2

                    ret = np.sum(di2, axis=1)
                    return ret

                dist_mat = make_dist_mat()

                # dist_mat special cases
                if self.use_categories:
                    dist_mat += (self.category_arr[
                                     index_array] - p_cat) ** 2  # compute distance of p to all patterns with category label
                if self.depression:
                    # dist_mat += self.depression_afun(self.depression_arr[index_array])       # Add additional distance based on previous usage
                    # dist_mat2 += self.depression_afun(self.depression_arr2)  # Add additional distance based on previous usage
                    dist_mat += self.depression_arr[
                        index_array]  # removed function usage because lambda is taking a lot of time
                    self.depression_arr *= np.exp(-1 / self.depression_recovery)  # Exponential decay

                min_dist = np.where(dist_mat == np.min(dist_mat))[0]  # get index/indices of minimum distance
                if len(min_dist) > 1:  # if multiple indices with minimum distance
                    ind_ind = np.argmin(np.abs(min_dist - last_index + 1))  # take the one with smallest jump from cue
                    ind = index_array[min_dist[ind_ind]]
                else:
                    ind = index_array[min_dist[0]]

            else:
                # make all calculations on latents instead, for debugging purposes
                if cue_from_keys:
                    lat_cue = self.lat_arr_keys[last_index]
                else:
                    lat_cue = self.lat_arr[last_index]
                nl = np.random.normal(0, self.retrieval_noise_level + 1e-20, len(lat_cue))
                plat = lat_cue+nl
                dist_mat = np.sum((self.lat_arr - plat)**2, axis=1)
                if self.use_categories:
                    dist_mat += (self.category_arr - p_cat) ** 2
                if self.depression:
                    dist_mat += self.depression_arr  # removed function usage because lambda is taking a lot of time
                    self.depression_arr *= np.exp(-1 / self.depression_recovery)  # Exponential decay

                min_dist = np.where(dist_mat == np.min(dist_mat))[0]  # get index/indices of minimum distance
                if len(min_dist) > 1:  # if multiple indices with minimum distance
                    ind_ind = np.argmin(np.abs(min_dist - last_index + 1))  # take the one with smallest jump from cue
                    ind = min_dist[ind_ind]
                else:
                    ind = min_dist[0]


            if self.depression:
                self.depression_arr[ind] += self.depression_use_cost

            if return_jumps:
                jumps.append(last_index + 1 - ind)

            if return_indices:
                indices_ret.append(self.lexicon_of_indices[ind])

            ret_pattern = self.pat_arr[ind]

            if self.return_err_values:
                if len(retrieved_sequence) > 0:
                    oridis = np.linalg.norm(self.helper_arr_keys[last_index] - self.helper_arr[last_index])
                    retdis = np.linalg.norm(ret_pattern - self.helper_arr[last_index])
                    errdis = retdis - oridis
                    error_distances.append(errdis)
                    if errdis == 0:
                        error_types.append(0)
                    elif np.array_equal(self.helper_arr_keys[last_index], self.helper_arr[(last_index + 1) % len(self.helper_arr)]):  # retrieval error was not forced by end of sequence
                        error_types.append(1)
                    else:
                        error_types.append(-1)

            retrieved_sequence.append(ret_pattern)  # retrieve pattern from memory
            cue = self.helper_arr_keys[ind]
            if self.use_categories:
                cue_category = self.category_arr_keys[ind]

            distance = np.sum((ret_pattern - cue)**2)

            if not self.optimization or distance < self.smoothing_threshold:
                last_index = ind
            else:
                cue = ret_pattern
                if self.use_categories:
                    cue_category = self.category_arr[ind]
                if np.array_equal(self.helper_arr_keys[ind - 1], ret_pattern):
                    last_index = ind - 1
                else:
                    last_index = -1
                # keyind = np.argmin(self.dmat[:, ind][np.nonzero(self.dmat[:, ind])])
                # cue = self.helper_arr_keys[keyind]
                # if self.use_categories:
                #     cue_category = self.category_arr_keys[keyind]
                # last_index = keyind

            retrieval_counter += 1
            if retrieval_counter == self.retrieval_length:
                break

        if return_opti_values:
            self.opti_values.append([10e10, np.shape(self.pat_arr)])

        if self.return_err_values:
            return np.array(retrieved_sequence), np.array(indices_ret), np.array(error_types), np.array(error_distances)
        if return_jumps:
            if return_indices:
                if return_opti_values:
                    return np.array(retrieved_sequence), np.array(jumps), np.array(indices_ret), self.opti_values
                return np.array(retrieved_sequence), np.array(jumps), np.array(indices_ret)
            if return_opti_values:
                return np.array(retrieved_sequence), np.array(jumps), self.opti_values
            return np.array(retrieved_sequence), np.array(jumps)
        if return_indices:
            if return_opti_values:
                return np.array(retrieved_sequence), np.array(indices_ret), self.opti_values
            return np.array(retrieved_sequence), np.array(indices_ret)
        if return_opti_values:
            return np.array(retrieved_sequence), self.opti_values
        return np.array(retrieved_sequence)