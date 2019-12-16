import numpy as np
import mdp
import random
import tools
import scipy.spatial.distance
import pdb
        
def store_sequence(self,episodic_memory,seq, categories=None, dimensions = 0):
    """Helper method in the case that masking off the first n dimensions of the sequence is important to do in one line."""
    sequence = seq  if dimensions < 1 else seq[:,:dimensions]
    episodic_memory.store_sequence(sequence, categories)
    
    
class EpisodicMemory():
    def __init__(self, p_dim, retrieval_length, retrieval_noise, category_weight=0,
                     weight_vector=1, completeness_weight = 0, depress_params=None):
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

    def store_sequence(self, seq, categories=None, completeness_scores=None):
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

        #preparation of retrieval
        retrieved_sequence = []      #initialize retrieved sequence

        if return_jumps:
            jumps = []                   #initialize array to hold jump size
        if return_indices:
            indices_ret = []         #initialize array to hold index for each returned sequence element

        if hasattr(cue_or_idx, '__iter__'):          #test if cue (array) or index of cue (integer) is given and get cue accordingly
            cue = cue_or_idx[:self.pattern_dim]
            if self.use_helper_array:
                cue = cue[self.helper_mask]
            last_index = -1                                         #initialize last index parameter, saving the index resulting from the last for iteration
        else:
            cue = self.helper_arr[cue_or_idx]
            last_index = cue_or_idx-1                    #cue_ind is determined later based on last_index+1

        retrieval_counter = 0

        def getCueIndex(cue, ind, usekeys=False):
            aoi = self.helper_arr if not usekeys else self.helper_arr_keys   #array of interest
            if ind < len(self.helper_arr) and np.array_equal(aoi[ind,:], cue):
                return ind
            #cue_ind_cand = np.unique(np.nonzero(aoi == cue)[0])   #this was wrong. a match in just one vector element was enough here. this happens rarely for float but it does.
            cue_ind_cand = np.nonzero((np.equal(aoi, cue)).all(axis=1))[0]
            if len(cue_ind_cand)==0:
                return -1            #return -1 if cue was not found in memory. This may happen when the end of a subsequence is reached, because the last element is omitted
            cue_ind_ind = np.argmin(abs(cue_ind_cand-ind))               #take cue index with smallest deviation from strict sequential retrieval
            cue_ind = cue_ind_cand[cue_ind_ind]
            return cue_ind

        #ACUTAL RETRIEVAL BEGINS HERE
        while True:
            if last_index == -1:  #last_index is -1 when retrieval_counter is 0
                last_index = getCueIndex(cue, last_index + 1, usekeys=True)
                #last_index is now still -1 if a first sequence element was given as cue

            n = np.random.normal(0,self.retrieval_noise_level+1e-20, self.helper_dim)
            p = cue + n

            if self.use_categories:
                nc = self.category_weight * random.gauss(0,self.retrieval_noise_level+1e-20)
                p_cat = cue_category + nc

            if last_index == -1:
                # when retrieval_counter is 0 and a first sequence element was given as cue
                last_index = getCueIndex(cue, -1) - 1  #analogously to the usual sequential structure we set last_index to one element before our cue pattern

            def make_dist_mat():
                di = self.helper_arr - p
                if self.do_retrieval_weighting:
                    mul = self.retrieval_weights*di
                else:
                    mul = di
                di2 = mul**2

                ret = np.sum(di2, axis=1)
                return ret
            dist_mat = make_dist_mat()

            # dist_mat special cases
            if self.use_categories:
                dist_mat += (self.category_arr - p_cat) ** 2   #compute distance of p to all patterns with category label
            if self.depression:
                #dist_mat += self.depression_afun(self.depression_arr)  # Add additional distance based on previous usage
                dist_mat += self.depression_arr  # removed function usage because lambda is taking a lot of time
                self.depression_arr *= np.exp(-1/self.depression_recovery)  # Exponential decay

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

            retrieved_sequence.append(self.pat_arr[ind])              #retrieve pattern from memory
            cue = self.helper_arr_keys[ind]

            if self.use_categories:
                cue_category = self.category_arr_keys[ind]

            last_index = ind

            retrieval_counter += 1
            if retrieval_counter == self.retrieval_length:
                break

        if return_jumps:
            if return_indices:
                return np.array(retrieved_sequence), np.array(jumps), np.array(indices_ret)
            return np.array(retrieved_sequence), np.array(jumps)
        if return_indices:
            return np.array(retrieved_sequence), np.array(indices_ret)
        return np.array(retrieved_sequence)