import numpy as np
import mdp
import random
import core.tools
import scipy.spatial.distance
import pdb
        
def store_sequence(self,episodic_memory,seq, categories=None, dimensions = 0):
    """Helper method in the case that masking off the first n dimensions of the sequence is important to do in one line."""
    sequence = seq  if dimensions < 1 else seq[:,:dimensions]
    episodic_memory.store_sequence(sequence, categories)
    
    
class EpisodicMemory():
    def __init__(self, p_dim, retrieval_length, retrieval_noise, category_weight=0,
                     weight_vector=1, depress_params=None):
        """Initialize episodic memory with given parameters
        :param p_dim: pattern dimensions
        :param ret_len: length of retrieved sequences. Must be given, but can be changed when calling retrieval function.
        :param ret_noise: standard deviation of retrieval noise. Must be given, but can be changed when calling retrieval function.
        :param categories: boolean defining whether memory should store category labels along with the sequence which are then used in retrieval. Default False
        """
        self.category_weight = category_weight
        self.pattern_dim = p_dim
        self.retrieval_length = retrieval_length
        self.retrieval_noise_level = retrieval_noise

        self.opti_values = []

        # if list is too short, zeros are appended, if too long, it is just cut off at the end. If single value, 
        # that is used for all features. By putting [1,1,1,1], only the first 4 features are considered for retrieval.
        if hasattr(weight_vector, '__iter__'):
            if len(weight_vector) >= self.pattern_dim:
                self.retrieval_weights = np.resize(np.array(weight_vector),self.pattern_dim)
                self.use_helper_array = (0 in self.retrieval_weights)
                        #helper array will be a copy of memory array, but elements corresponding to 0-weights will be removed. Speeds up calculation.
            else:
                self.retrieval_weights = np.array(weight_vector)
                self.use_helper_array = True
        else:
            self.retrieval_weights = np.array([weight_vector]*self.pattern_dim)
            self.use_helper_array = False
        #for memory and category a list and an ndarray is stored because appending of sequences is way faster for lists,
        #but vector-calculations are only efficient on ndarrays.
        self.memory_list = []      #actual episodic memory as list of patterns for storage
        self.memory_length = 0  #cumulative length of sequences in memory
        self.lexicon_of_indices = []  #the mighty look-up list of indices - to translate internal indices to total sequence indices. Because we remove the last element of each sequence.
        if self.use_helper_array:
            self.helper_mask = self.retrieval_weights>0
            self.helper_dim = len(np.where(True==self.helper_mask)[0])
            self.do_retrieval_weighting = not (1 == len(np.unique(self.retrieval_weights[self.helper_mask]))) #if retrieval weights are binary, which is most likely, weighting during retrieval is not neccessary
            self.helper_list = []
        else:  # if helper array is not used, it is set to reference to memory array - so memory array is used for retrieval
            self.helper_dim = self.pattern_dim
            self.do_retrieval_weighting = not (1 == len(np.unique(self.retrieval_weights)))
            self.helper_list = self.memory_list
        self.retrieval_weights = self.retrieval_weights[np.where(self.retrieval_weights != 0)]  # remove 0 elements because they will not be considered in helper array

        self.depression = depress_params is not None
        if self.depression :
            self.depression_arr = np.empty( (0, 1) )
            self.depression_use_cost = depress_params['cost']
            self.depression_afun = eval(depress_params.get('activation_function', 'lambda X: X'))    #CARE: afun is not used in current implementation
            self.depression_recovery = float(depress_params['recovery_time_constant'])

        self.changed = False          #keeps track whether memory_arr matches memory_list or if a conversion is necessary
        self.use_categories = self.category_weight > 0
        if self.use_categories:
            self.category_list = []  # list of category labels, one label corresponds to one frame

    def store_sequence(self, seq, categories=None):
        """Store sequence in episodic memory. Create tuple of (pattern, next_pattern) for each element in sequence.
        :param seq: sequence to store (ndarray)
        :param categories: optional. category sequence to store (ndarray), one label for each pattern in seq.
        """
        if not len(seq[0]) == self.pattern_dim:
            raise Exception("pattern dimension mismatch")
        if self.use_categories:
            if categories is None:
                raise Exception("Memory is set so store categories, but None given.")
            self.category_list.append(categories)
        
        self.memory_list.append(seq)     #add to list of arrays
        if self.use_helper_array:
            self.helper_list.append(np.array([pat[self.helper_mask] for pat in seq]))  #pat as output op mdp flow should be numpy array, so list indexing is possible
        self.lexicon_of_indices.extend([x+self.memory_length for x in range(len(seq)-1)])   #beginning at the index for the case without omitting last element, append increasing index sequence.
                                                            #appended index sequence is 1 shorter than original sequence, because later in retrieval, last element of each sequence is omitted.
        self.memory_length = self.memory_length+len(seq)
        self.changed = True
    
    def retrieve_sequence(self, cue_or_idx, cue_category=None, ret_len = None, ret_noise = None, return_jumps=False, return_indices=False):
        """Retrieve sequence from episodic memory given a cue.
        :param cue_or_idx: pattern as ndarray with correct number of dimensions, or index of cue to use from memory. Note that if a pattern is given, this pattern actually has to be taken from the stored sequences!
        :param cue_category: optional. category of given cue as {0,1}.
        :param ret_len: length of retrieved sequence. If None, keep former setting. Default None.
        :param ret_noise: standard deviation of retrieval noise. If None, keep former setting. Default None.
        :param return_jumps: if True, function returns tuple (sequence, jumps, [indices]). jumps contains the distance of the retrieved pattern from the cue for each sequence element. 0 for no error. Default False.
        :param return_indices: if True, function returns tuple (sequence, [jumps], indices). indices contains the index within memory for each retrieved pattern of the sequence. Default False.
        :returns: retrieved sequence as ndarray
        """
        # setup parameters and make memory arrays
        if ret_len is not None:
            self.retrieval_length = ret_len    #if given, overwrite retrieval parameters
        if ret_noise is not None:
            self.retrieval_noise_level = ret_noise
        if self.changed:                                                       #if list was changed,
            self.changed = False
            if len(self.memory_list) == 1:
                np.append(self.memory_list[0],self.memory_list[0][-1])               #if only one sequence was stored, we want to detect if we reach the end of the sequence, so we need to include last element
                if self.use_helper_array:
                    np.append(self.helper_list[0], self.helper_list[0][-1])
                self.lexicon_of_indices.append(self.lexicon_of_indices[-1])         #then we also need one more element in the mighty look-up list of indices

            self.pat_arr = np.concatenate([seq[:-1] for seq in self.memory_list])  #memory of full patterns, last element is not included. It appears only as key paired with the second-to-last pattern
            if self.use_helper_array:
                self.helper_arr = np.concatenate([seq[:-1] for seq in self.helper_list])    #memory of incomplete patterns, according to weight-vector
                self.helper_arr_keys = np.concatenate([seq[1:] for seq in self.helper_list])    #incomplete keys. The first element of each sequence is omitted since there is no previous element to it
            else:
                self.helper_arr = self.pat_arr
                self.helper_arr_keys = np.concatenate([seq[1:] for seq in self.memory_list])  # full memory keys
            if self.use_categories:
                if cue_category is None:
                    raise Exception("Memory is set to store categories, but no category for cue given.")
                if len(self.memory_list) == 1:
                    np.append(self.category_list[0], self.category_list[0][-1])        #only one sequence, repeat last element, as for sequences above
                self.category_arr = self.category_weight*np.concatenate([cat_list[:-1] for cat_list in self.category_list])      #as it happens for sequences, remove last element from cat array
                self.category_arr_keys = self.category_weight*np.concatenate([cat_list[1:] for cat_list in self.category_list])

            if self.depression:
                self.depression_arr = np.zeros(self.helper_arr.shape[0])
                self.depression_arr2 = np.zeros(self.helper_arr.shape[0])

            if self.do_retrieval_weighting:
                if self.use_categories:
                    self.dmat = scipy.spatial.distance.cdist(
                        np.append(self.retrieval_weights*self.helper_arr_keys, self.category_arr_keys[:, None], axis=1),
                        np.append(self.retrieval_weights*self.helper_arr, self.category_arr[:, None], axis=1),
                        'sqeuclidean')  # SQUARED distances of all keys to all patterns with category
                else:
                    self.dmat = scipy.spatial.distance.cdist(self.retrieval_weights * self.helper_arr_keys,
                                                             self.retrieval_weights * self.helper_arr,
                                                             'sqeuclidean')  # and without cat
            else:
                if self.use_categories:
                    self.dmat = scipy.spatial.distance.cdist(
                        np.append(self.helper_arr_keys, self.category_arr_keys[:,None], axis=1),
                        np.append(self.helper_arr, self.category_arr[:,None], axis=1),
                        'sqeuclidean')  # SQUARED distances of all keys to all patterns with category
                else:
                    self.dmat = scipy.spatial.distance.cdist(self.helper_arr_keys,self.helper_arr,'sqeuclidean') #and without cat
            self.dsort = np.argsort(self.dmat, axis=1)      #sorted indices for each key-patterns distance vector

        #preparation of retrieval
        retrieved_sequence = []      #initialize retrieved sequence
        retrieved_sequence2 = []  # initialize retrieved sequence

        if return_jumps:
            jumps = []                   #initialize array to hold jump size
            jumps2 = []  # initialize array to hold jump size
        if return_indices:
            indices_ret = []         #initialize array to hold index for each returned sequence element
            indices_ret2 = []  # initialize array to hold index for each returned sequence element

        if hasattr(cue_or_idx, '__iter__'):          #test if cue (array) or index of cue (integer) is given and get cue accordingly
            cue = cue2= cue_or_idx[:self.pattern_dim]
            if self.use_helper_array:
                cue = cue2 = cue[self.helper_mask]
            last_index = last_index2 = -1                                         #initialize last index parameter, saving the index resulting from the last for iteration
        else:
            cue = cue2 = self.helper_arr[cue_or_idx]
            last_index = last_index2 = cue_or_idx-1                    #cue_ind is determined later based on last_index+1

        retrieval_counter = 0

        def getCueIndex(cue, ind, cat_cue=None, usekeys=False):
            aoi = self.helper_arr if not usekeys else self.helper_arr_keys   #array of interest
            if cat_cue is not None:
                coi = self.category_arr if not usekeys else self.category_arr_keys  # category_array of interest
                if ind < len(self.helper_arr) and np.array_equal(aoi[ind,:], cue) and coi[ind] == cat_cue:
                    return ind
            elif ind < len(self.helper_arr) and np.array_equal(aoi[ind,:], cue):
                return ind
            #cue_ind_cand = np.unique(np.nonzero(aoi == cue)[0])   #this was wrong. a match in just one vector element was enough here. this happens rarely for float but it does.
            if cat_cue is not None:
                cue_ind_cand = np.nonzero((np.equal(np.append(aoi,coi[:,None],axis=1), np.append(cue,cat_cue))).all(axis=1))[0]
            else:
                cue_ind_cand = np.nonzero((np.equal(aoi, cue)).all(axis=1))[0]
            if len(cue_ind_cand)==0:
                return -1            #return -1 if cue was not found in memory. This may happen when the end of a subsequence is reached, because the last element is omitted
                #returning -1 instead of 0 because this way the jump can be set to 0 manually. Otherwise this may result in really unrealistic jump sizes.
                #-1 is unproblematic here, because cue_ind is never used as an actual index to access data.
            cue_ind_ind = np.argmin(abs(cue_ind_cand-ind))               #take cue index with smallest deviation from strict sequential retrieval
            cue_ind = cue_ind_cand[cue_ind_ind]
            return cue_ind

        if self.use_categories:
            if cue_category:
                cue_category = self.category_weight
            else:
                cue_category = 0
            cue_category2 = cue_category

        #ACUTAL RETRIEVAL BEGINS HERE
        while True:
            if last_index == -1:  #last_index is -1 when retrieval_counter is 0
                if self.use_categories:
                    last_index = getCueIndex(cue, last_index + 1, cue_category, usekeys=True)
                else:
                    last_index = getCueIndex(cue, last_index + 1, usekeys=True)
                #last_index is now still -1 if a first sequence element was given as cue

            n = np.random.normal(0,self.retrieval_noise_level+1e-20, self.helper_dim)
            p = cue + n
            p2 = cue2 + n

            if self.use_categories:
                nc = self.category_weight * random.gauss(0,self.retrieval_noise_level+1e-20)
                p_cat = cue_category + nc
                p_cat2 = cue_category2 + nc

            if not last_index == -1:   #last index can be -1 only in the first iteration, if a cue was chosen that does not exist as key (first sequence element)
                d0 = self.dmat[last_index, self.dsort[last_index, 0]]  # most probably 0, but not when key at last_index is one of the omitted last sequence elements
                nn = np.append(n, nc) if self.use_categories else n
                two_n_plus_d0 = d0 + 4*np.sum(nn**2)    #4 times - usually we would need two for two times the distance, but since we are using squared distances we need 4
            else:
                #when retrieval_counter is 0 and a first sequence element was given as cue
                #the cue is probably similar to its associated key, so we look for the index
                if self.use_categories:
                    last_index = getCueIndex(cue, -1, cue_category) #this time, last_index is actually not the last index
                else:
                    last_index = getCueIndex(cue, -1)   #same here
                cue_corr_key = self.helper_arr_keys[last_index]
                # last_index ist used later twice:
                # 1. selecting row from dsort, here we just need it to select the row corresponding to cue_corr_key
                # 2. if several candidates with minimum distance are found, select the closest, no problem because cue anyway was not in memory as key
                dis = np.sum((cue_corr_key-cue)**2)
                d0 = self.dmat[last_index, self.dsort[last_index, 0]]
                nn = np.append(n, nc) if self.use_categories else n
                two_n_plus_d0 = d0 + 4 * np.sum(nn ** 2) + 4*dis     #4 times since we are using squared distances

            if last_index2 == -1:
                last_index2 = last_index

            if self.depression:
                two_n_plus_d0 += 4*np.abs(np.max(self.depression_arr))

            def findsmart(dma, dso, d):
                """given a number d and an unsorted distance vector dma and an accordingly argsorted index-vector dso
                finding the index of dso with the largest value in dma that is smaller as or equal to d."""
                l = len(dso)
                if l == 1:
                    return 0
                ii = l/2
                if dma[dso[ii]] > d:
                    return findsmart(dma,dso[:ii],d)
                else:
                    return ii + findsmart(dma,dso[ii:],d)
            index_range = 1+findsmart(self.dmat[last_index, :], self.dsort[last_index, :], two_n_plus_d0) #function returns index, so add one to use as range boundary

            self.opti_values.append([two_n_plus_d0, index_range])

            index_array = self.dsort[last_index, :index_range]
            part_arr = self.helper_arr[index_array]     #take only part of the memory into account, according to index_array

            def make_dist_mat():
                di = part_arr - p
                if self.do_retrieval_weighting:
                    mul = self.retrieval_weights*di
                else:
                    mul = di
                di2 = mul**2

                ret = np.sum(di2, axis=1)
                return ret
            dist_mat = make_dist_mat()

            def make_dist_mat2():
                di = self.helper_arr - p2
                if self.do_retrieval_weighting:
                    mul = self.retrieval_weights*di
                else:
                    mul = di
                di2 = mul**2

                ret = np.sum(di2, axis=1)
                return ret
            dist_mat2 = make_dist_mat2()

            # dist_mat special cases
            if self.use_categories:
                dist_mat += (self.category_arr[index_array] - p_cat)**2 #compute distance of p to all patterns with category label
                dist_mat2 += (self.category_arr - p_cat2) ** 2
            if self.depression:
                #dist_mat += self.depression_afun(self.depression_arr[index_array])       # Add additional distance based on previous usage
                #dist_mat2 += self.depression_afun(self.depression_arr2)  # Add additional distance based on previous usage
                dist_mat += self.depression_arr[index_array]  # removed function usage because lambda is taking a lot of time
                dist_mat2 += self.depression_arr2
                self.depression_arr *= np.exp(-1/self.depression_recovery)  # Exponential decay
                self.depression_arr2 *= np.exp(-1 / self.depression_recovery)  # Exponential decay

            if not np.array_equal(dist_mat, dist_mat2[index_array]):
                pdb.set_trace()
                raise Exception("Ich kann leider nicht programmieren")
            
            #dist_mat[:] = np.sqrt(dist_mat)
    
            min_dist = np.where(dist_mat == np.min(dist_mat))[0]               # get index/indices of minimum distance
            if len(min_dist) > 1:                                           # if multiple indices with minimum distance
                ind_ind = np.argmin(np.abs(min_dist-last_index+1))                  # take the one with smallest jump from cue
                ind = index_array[min_dist[ind_ind]]
            else:
                ind = index_array[min_dist[0]]

            min_dist2 = np.where(dist_mat2 == np.min(dist_mat2))[0]  # get index/indices of minimum distance
            if len(min_dist2) > 1:  # if multiple indiforming_categoriesces with minimum distance
                ind_ind2 = np.argmin(np.abs(min_dist2 - last_index2 + 1))  # take the one with smallest jump from cue
                ind2 = min_dist2[ind_ind2]
            else:
                ind2 = min_dist2[0]
            
            if self.depression:
                self.depression_arr[ind] += self.depression_use_cost
                self.depression_arr2[ind2] += self.depression_use_cost

            if return_jumps:
                jumps.append(last_index + 1 - ind)
                jumps2.append(last_index2 + 1 - ind2)

            if return_indices:
                indices_ret.append(self.lexicon_of_indices[ind])
                indices_ret2.append(self.lexicon_of_indices[ind2])

            li2 = last_index2 if not last_index2 == -1 else last_index
            diffout = dist_mat2 - self.dmat[li2, :]
            #d00 = self.dmat[li2, self.dsort[li2, 0]]  # most probably 0, but not when key at last_index is one of the omitted last sequence elements
            #two_n_plus_d00 = d00 + 2 * np.linalg.norm(nn)**2
            nabs = np.abs(n)
            cueabs = np.abs(cue)
            if not ind == ind2:
                print("")
                print("==========DEBUG OUTPUT=============")
                print("retrieval counter: {}".format(retrieval_counter))
                print("last index: {}".format(li2))
                print("best guess for ind (index of lowest in dmat): {}".format(self.dsort[li2, 0]))
                print("ind: {}".format(ind))
                print("ind2: {}".format(ind2))
                print("Position of ind2 in dsort: {}".format(np.where(self.dsort[li2, :] == ind2)[0]))
                print("'Circle radius' in pattern space: {}".format(two_n_plus_d0))
                print("Length of dist_mat (index_range): {}".format(index_range))
                print("Maximum / minimum (element-wise) difference between dmat and dist_mat2: {} / {}".format(np.max(diffout), np.min(diffout)))
                print("Difference between dmat and dist_mat2 at ind / at ind2: {} / {}".format(diffout[ind], diffout[ind2]))
                print("dmat at ind / at ind2: {} / {}".format(self.dmat[li2, ind], self.dmat[li2, ind2]))
                print("Lowest distance in dmat (usually 0): {}".format(d0))
                print("Highest distance in dmat: {}".format(self.dmat[li2, self.dsort[li2, -1]]))
                print("Min distance in dist_mat: {}".format(np.min(dist_mat)))
                print("Min distance in dist_mat2: {}".format(np.min(dist_mat2)))
                print("Noise size: {}".format(np.sum(n ** 2)))
                print("===================================")
                out_of_bounds_difference = self.dmat[li2, ind2] - two_n_plus_d0
                out_of_bounds_fraction = self.dmat[li2, ind2] / two_n_plus_d0
                return out_of_bounds_difference, out_of_bounds_fraction
                #pdb.set_trace()
                raise Exception("Dinge sind schief gelaufen")

            retrieved_sequence.append(self.pat_arr[ind])              #retrieve pattern from memory
            retrieved_sequence2.append(self.pat_arr[ind2])
            cue = self.helper_arr_keys[ind]
            cue2 = self.helper_arr_keys[ind2]

            if self.use_categories:
                cue_category = self.category_arr_keys[ind]
                cue_category2 = self.category_arr_keys[ind2]

            last_index = ind
            last_index2 = ind2

            retrieval_counter += 1
            if retrieval_counter == self.retrieval_length:
                break

        self.opti_values.append([10e10, np.shape(self.pat_arr)])

        if return_jumps:
            if return_indices:
                return np.array(retrieved_sequence), np.array(jumps), np.array(indices_ret), self.opti_values
            return np.array(retrieved_sequence), np.array(jumps), self.opti_values
        if return_indices:
            return np.array(retrieved_sequence), np.array(indices_ret), self.opti_values
        return np.array(retrieved_sequence), self.opti_values