"""
Contains the :py:class:`SysParamSet` class that holds all parameters required for a simulation run.

"""

from . import input_params
from . import semantic_params


class SysParamSet():

    # def __getstate__(self):
    #     return {key:getattr(self,key) for key in self.__dict__ if key is not "input_params_default"}

    def __init__(self):
        """
        An object of this class contains all the parameters required to run a simulation with
        :py:func:`core.streamlined.program`. A program run of *streamlined* has 4 stages:

        * **stage 1**: generate training data, create sfa1, train sfa1
        * **stage 2**: create sfa2, create EpisodicMemory, generate forming and testing data, execute sfa1 on forming data
        * **stage 3**: store sfa1 features in EpisodicMemory, retrieve sequences, train sfa2s
        * **stage 4**: run testing data through sfa1 and sfa2s. If enabled, make test EpisodicMemories and
          use them to evaluate feature quality of sfa2s output.

        The parameters are also structured by stages, but it is not done very consistently.
        See the docstrings for the respective attributes for more info:

        * **stage 1**: :py:data:`st1`
        * **stage 2**: :py:data:`st2`
        * **stage 3**: :py:data:`st3`
        * **stage 4**: :py:data:`st4`

        In addition to the stages, general parameters are defined that are not associated with any stage in particular.

        """
        self.data_description = ""
        """| default: ``''``
           | Just a description for data saved in results file"""

        self.result_path = ""
        """| default: ``''``
           | where to store result files. Is actually only used by *streamlined* when set to repeat SFA2 training multiple times
           | (cf. :py:data:`st3`). In that case, SFA2 modules are pickled into that path."""

        self.filepath_sfa1 = "results/SFA1.p"
        """| default: ``'results/SFA1.p'``
           | where to load sfa1 from and where to store to. Is only relevant if :py:data:`load_sfa1` and/or
           | :py:data:`save_sfa1` are True.
        """

        self.load_sfa1 = False
        """| default: ``False``
           | whether to load sfa1 from file instead of creating and training it. Path is given by py:data:`core.system_params.filepath_sfa1`
        """

        self.save_sfa1 = False
        """| default: ``False``
           | whether to save the trained sfa1 to a file. Path is given by py:data:`core.system_params.filepath_sfa1`.
        """

        self.normalization = "whiten.ZCA"
        """| default: ``'whiten.ZCA'``
           | Method of normalization/whitening of SFA output. Can be ``'scale'``, ``'whiten'``, ``'whiten.ZCA'``, or ``'none'``
        """

        self.generate_debug_presequences = True
        """| default: ``True``
           | Whether or not to generate the *retrieved_presequence* and *perfect_presequence* data after retrieval from EpisodicMemory.
           | *retrieved_presequence* is a sequence of forming input images that corresponds to the sequence of SFA1 features retrieved
           | from memory. This way the sequence retrieved from memory can somewhat be visualized. *perfect_presequence* is similar, but
           | without retrieval errors or jumps at the end of an episode. That means these are the forming input images that correspond
           | to the sequences of SFA1 features that follow in memory after the retrieval cue.
        """

        self.same_input_for_all = False
        """| default: ``False``
           | If True, training input is generated once according to the parameters in :py:data:`st1`
           | and the exact same input data is used for the other stages, that means no additional forming or testing input is 
           | generated.
        """

        self.which = 'ES'
        """| default: ``'ES'``
           | Which sfa modules to create and train. If S is contained in the string, sfa2S is taken care of, same for E and sfa2E.
        """
        self.program_extent = 4
        """| default: ``'ES'``
           | Which sfa modules to create and train. If S is contained in the string, sfa2S is taken care of, same for E and sfa2E.
        """

        self.sem_params1 = semantic_params.make_jingnet(16)
        """| default: ``semantic_params.make_jingnet(16)``
           | Parameter list for SFA1
           | See :py:mod:`core.semantic_params` for more information
        """

        self.sem_params2E = None
        """| default: ``semantic_params.make_layer_series(16,16,20,20,16,16)``
           | Parameter list for SFA2E
        """

        self.sem_params2S = None
        """| default: ``semantic_params.make_layer_series(16,16,20,20,16,16)``
           | Parameter list for SFA2S
        """

        self.setsem2(semantic_params.make_layer_series(16,16,20,20,16,16))

        self.input_params_default = input_params.catsndogs
        """| default: ``input_params.catsndogs``
           | Default parameters for input generation. Parts of these are overridden in the stage dictionaries
           | (:py:data:`st1`, :py:data:`st2`, :py:data:`st4`)
           | See :py:mod:`core.input_params` for more information
        """

        self.st1 = {
            'number_of_snippets': 200,  # number of individual snippets to generate
            'snippet_length': None,     # number of frames of each input episode
        }
        """default::
           
              {
               'number_of_snippets': 200,  # number of individual snippets to generate
               'snippet_length': None,     # number of frames of each input episode
              }
        
        Stage 1 is about constructing the first SFA, which generates an incomplete feature
        representation. All parameters in :py:data:`input_params_default` can
        be overridden here.
        """

        # STAGE 2 --- here, we generate new input, which we call "selection" as opposed to the "training"
        # data in stage 1. This term arises because of historical reasons (we used to select a subsection
        # of our st1 training data here), but we're keeping it because the name is nicely distinct from
        # the usual dichotomy of "testing/training" which makes the purpose easier to remember/comprehend.
        # Then feed new input through SFA1 to get input data for SFA2. In the episodic case, first store and
        # retrieve in episodic memory to make things smoother.
        self.st2 = {
            # ----- overriding input_params_default -----
            'number_of_snippets': 50,
            'input_noise': 0.1,
            # ----- specific to streamlined ----
            'sfa2_noise': 0,
            'number_of_retrieval_trials': 200,
            'memory': {
                # ----- EpisodicMemory parameters ----
                'category_weight': 0,
                'retrieval_length': 200,
                'retrieval_noise': 0.02,
                'weight_vector': 1,
                'smoothing_percentile': 100,
                'optimization': True,
                'use_latents': False,
                'return_err_values': False
            }}
        """default::
        
              {
              # ----- overriding input_params_default -----
              'number_of_snippets': 50,
              'input_noise': 0.1,
              # ----- specific to streamlined ----
              'sfa2_noise': 0,  # std of Gaussian noise added to both training data sets of sfa2S and sfa2E (after retrieval from EpisodicMemory!)
              'number_of_retrieval_trials': 200, # how many sequences to retrieve from EpisodicMemory
              'memory': {
                   # ----- EpisodicMemory parameters ----
                   'category_weight': 0,
                   'retrieval_length': 200,
                   'retrieval_noise': 0.02,
                   'weight_vector': 1,
                   'optimization': True,
                   'use_latents': False,
                   'return_err_values': False
              }}

           Stage 2 is about generating forming data and creating the EpisodicMemory.
           All parameters in :py:data:`input_params_default` can
           be overridden here (top part of default dictionary).
           
           Additionally, there are two parameters specific to *streamlined*,
           namely ``sfa2_noise'`` and ``'number_of_retrieval_trials'`` (middle part of default dictionary).
           See the comments in the literal block for explanation.
           This is where the inconsistency begins, because retrieval is actually done
           in stage 3, while this parameter is here in st2. Deal with it.
           
           st2 also contains the ``'memory'`` dictionary which contains input arguments of the
           constructor for the class :py:class:`core.episodic.EpisodicMemory`.
               """

        self.st3 = {
            'inc_repeats_S': 1,     # if > 1, training of sfa2S is repeated this number of times (if sfa2S is incremental)
            'inc_repeats_E': 1,     # same for sfa2E
            'retr_repeats': 1,      # if > 1, episodic memory retrieval and subsequent training of sfa2E is repeated this number of times (if sfa2E is incremental)
                                    # this is independent of 'inc_repeats', which repeats training on the last retrieved set of sequences.
            'cue_equally': False,   # make sure that each object type is used as a retrieval cue the same number of times
            'learnrate': 0.001,     # learnrate of incremental sfa2
            'use_memory': True      # whether to do retrieval (True) or to set retrieved_sequence to forming_sequence (False)
        }
        """default::
        
              { 
              'inc_repeats_S': 1,     # if > 1, training of sfa2S is repeated this 
                                      # number of times (if sfa2S is incremental)
                                      
              'inc_repeats_E': 1,     # same for sfa2E
              
              'retr_repeats': 1,      # if > 1, episodic memory retrieval and subsequent
                                      # training of sfa2E is repeated this number of times 
                                      # (if sfa2E is incremental)
                                      # this is independent of 'inc_repeats', which repeats
                                      # training on the last retrieved set of sequences.
                                      
              'cue_equally': False,   # make sure that each object type is used as a 
                                      # retrieval cue the same number of times
                                      
              'learnrate': 0.001,     # learnrate of incremental sfa2
              
              'use_memory': True      # whether to do retrieval (True) or to set retrieved_sequence
                                      # to forming_sequence (False)
              }    
        
        Stage 3 is about storing and retrieving episodic memories and training sfa2. See the comments in the
        literal block for explanation of the parameters. If ``'inc_repeats_S'`` and/or
        ``'inc_repeats_E'`` and/or ``'retr_repeats'`` are > 0, the SFA module is pickled to the path given by :py:data:`result_path`
        after each training repetition.
        """

        self.st4 = {
            'number_of_snippets': 50,
            'input_noise': 0.1,
            'sfa2_noise': 0,       # std of Gaussian noise added to sfa1 output before feeding it through sfa2S and sfa2E
            'do_memtest': False,   # whether or not generate fresh EpisodicMemory objects and store and retrieve sfa2S and sfa2E output to compare retrieval errors
            'number_of_retrievals': 50,  # number of sequences to retrieve from the test EpisodicMemory
            'memtest': {
                'weight_vector': 1,
                'retrieval_length': 50,
                'retrieval_noise': 0.02,
                'category_weight': 0,
                'optimization': True}
        }
        """default::

              {
              # ----- overriding input_params_default -----
              'number_of_snippets': 50,
              'input_noise': 0.1,
              # ----- specific to streamlined ----
              'sfa2_noise': 0,              # std of Gaussian noise added to sfa1 
                                            # output before feeding it through sfa2S and sfa2E
              'do_memtest': False,          # whether or not generate fresh EpisodicMemory objects and store and 
                                            # retrieve sfa2S and sfa2E output to compare retrieval errors
              'number_of_retrievals': 50,   # number of sequences to retrieve from the test EpisodicMemory
              'memtest': {
                  # ----- EpisodicMemory parameters ----
                  'weight_vector': 1,
                  'retrieval_length': 50,
                  'retrieval_noise': 0.02,
                  'category_weight': 0,
                  'optimization': True}  

        Stage 4: testing. This dictionary is similar to that of stage 2 (:py:data:`st2`).
        What is inconsistent here is that testing data is actually 
        generated in stage 2 in streamlined. However, the st4 dict contains, among others, parameters for testing input
        generation. All parameters in :py:data:`input_params_default` can
        be overridden here (top part of default dictionary).
        
        Additionally, there are three parameters specific to *streamlined*, (middle part of default dictionary). See 
        the comments in the literal block for explanation.
        
        Also, the ``'memtest'`` dictionary is defined which contains input arguments of the
        constructor for the class :py:class:`core.episodic.EpisodicMemory`. Only relevant if ``'do_memtest' = True'``.
        """

        self.st4b = None
        """default: ``None``
        
        This can be a second set of testing data if not None. It is used the same way as the first set and evaluated seperately.
        """

    def setsem2(self, semparams):
        """
        Helper method to set parameters for both sfa2S and sfa2E
        (*sem_params2S* and *sem_params2E*, respectively) to the same value

        :param semparams: sfa parameter list (cf. :py:mod:`core.semantic_params`)
        """
        self.sem_params2S = semparams
        self.sem_params2E = semparams

    def get(self, name):
        """
        returns the value of the parameter in question, going to defaults
        if not explicitly declared in a stage variable.

        Example usage: ``paramset.get('st2.snippet_length')``

        :param name: name of requested parameter
        :return: value of parameter in this object or in default input parameters, if not found here
        """
        parts = name.split('.', 1)
        return getattr(self, parts[0]).get(parts[1], self.input_params_default[parts[1]])