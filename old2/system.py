from core import semantic, episodic, sensory
from core import system_params, tools, result, input_params

import numpy as np
import random
import sys

if __name__ == "__main__":
    PARAMETERS = system_params.SysParamSet()
    LOAD_SFA1, SAVE_SFA1 = (PARAMETERS.load_sfa1, PARAMETERS.save_sfa1)
    FILEPATH_SFA1 = PARAMETERS.filepath_sfa1

    PARAMETERS.st2['object_code'] = input_params.make_object_code('ZB')
    PARAMETERS.st2['sequence'] = [0, 1]

    PARAMETERS.st2['movement_type'] = 'stillframe'
    PARAMETERS.st2["movement_params"] = dict()
    PARAMETERS.st2["number_of_snippets"] = 5000
    PARAMETERS.st2["snippet_length"] = 2
    PARAMETERS.st2["interleaved"] = True
    PARAMETERS.st2["blank_frame"] = False
    PARAMETERS.st2["glue"] = "random"

    PARAMETERS.st2['memory']['category_weight'] = 5
    PARAMETERS.st2['memory']['retrieval_noise'] = 0.2

    PARAMETERS.st4['object_code'] = input_params.make_object_code('XO')
    PARAMETERS.st4['sequence'] = [0, 1]
    PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
    PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=50,
                                             border_extent=2.3)
    PARAMETERS.st4["number_of_snippets"] = 50
    PARAMETERS.st4["snippet_length"] = None
    PARAMETERS.st4["interleaved"] = True
    PARAMETERS.st4["blank_frame"] = False
    PARAMETERS.st4["glue"] = "random"
        
    if PARAMETERS.st4['output_to_file']:
        #get running index
        f = open(".ind", "r")
        running_ind = int(f.readline())
        f.close()
        fw = open(".ind", "w")
        fw.write(str(running_ind+1))
        fw.close()

    #=========INITIALIZATION OF SYSTEMS======================================
    semantic_system = semantic.SemanticSystem()
    
    if LOAD_SFA1:
        sys.stdout.write("Loading SFA1.. ")
        sys.stdout.flush()
        sfa1 = semantic_system.load_SFA_module(FILEPATH_SFA1)
        print("done")
    else:
        sfa1 = semantic_system.build_module(PARAMETERS.sem_params1)
    sfa2S = semantic_system.build_module(PARAMETERS.sem_params2S)
    sfa2E = semantic_system.build_module(PARAMETERS.sem_params2E)

    sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default)
    
    memory   = episodic.EpisodicMemory( sfa1[-1].output_dim, categories=True, **PARAMETERS.st2['memory'])
    memtestS = episodic.EpisodicMemory(sfa2S[-1].output_dim, **PARAMETERS.st4['memtest'] )
    memtestE = episodic.EpisodicMemory(sfa2E[-1].output_dim, **PARAMETERS.st4['memtest'] )
    
    print("Initialized all systems")
    print("STAGE 1")
    sys.stdout.write("Generating input.. ")
    sys.stdout.flush()
        
    #=========MUTUAL STAGE 1: BASIC INPUT GENERATION, SFA1 TRAINING==========
    training_sequence, training_categories,training_latent = sensory_system.generate(**PARAMETERS.st1)
    selection_sequence, selection_categories, selection_latent, selection_ranges = sensory_system.generate(fetch_indices=True,**PARAMETERS.st2)
    testing_sequence, testing_categories, testing_latent = sensory_system.generate(**PARAMETERS.st4)
    
    #tools.preview_input(training_sequence)
            
    print("done")
    if not LOAD_SFA1:
        sys.stdout.write("Training SFA1.. ")
        sys.stdout.flush()
        #SFA1 training with long training sequence
        semantic.train_SFA(sfa1, training_sequence)
        print("done")
    
    print(semantic.get_d_values(sfa1))
    
    if SAVE_SFA1:
        sys.stdout.write("Saving SFA1.. ")
        sys.stdout.flush()
        sfa1.save(FILEPATH_SFA1)
        print("done")
    
    print("STAGE 2")
    sys.stdout.write("Randomly sub-selecting training input snippets.. ")
    sys.stdout.flush()
    
    #=========MUTUAL STAGE 2: SUBSELECTION OF TRAINING INPUT, RUN SFA1=======
    #random selection of training sequences and concatenation
#     selection = np.random.random_integers(0,PARAMETERS.st1['number_of_snippets']-1,PARAMETERS.st2['number_subselection'])
#     selection_sequence, selection_categories, selection_latent = sensory_system.recall(selection,**PARAMETERS.st2)
    
    #tools.preview_input(selection_sequence)

    print("done")
    sys.stdout.write("Generating features by running SFA1 on snippets.. ")
    sys.stdout.flush()
    
    #run SFA1 on randomly selected training data 
    SFA1_output = semantic_system.run_SFA(sfa1, selection_sequence)
    
    print("done")
    print("Storing features in memory and generating scenarios.. "),
    #=========EPISODIC STAGE 2: STORE AND RETRIEVE SFA1 OUTPUT===============
    for ran in selection_ranges:
        liran = list(ran)
        memory.store_sequence(SFA1_output[liran], categories=selection_categories[liran])
 
    stage2_retrievals_list = []
    stage2_visual = []
    comparison_visual = []
    BLACK = np.ones((30, 30)) * 255
    for j in range(PARAMETERS.st2['number_of_retrieval_trials']):
        cue_index = random.randint(0,len(SFA1_output)-1)
        #retrieved_sequence = memory.retrieve_sequence(SFA1_output[cue_index], selection_categories[cue_index], PARAMETERS.st2.retrieval_length, PARAMETERS.st2.retrieval_noise)
        retrieved_sequence, retrieved_indices = memory.retrieve_sequence(SFA1_output[cue_index], selection_categories[cue_index],return_indices=True)
        stage2_retrievals_list.append(retrieved_sequence)
        comparison_visual.extend(selection_sequence[cue_index:cue_index + PARAMETERS.st2['memory']['retrieval_length']])
        comparison_visual.extend([BLACK] * 5)
        stage2_visual.extend(selection_sequence[retrieved_indices])
        stage2_visual.extend([BLACK] * 5)
    stage2_retrieval_arr = np.concatenate(stage2_retrievals_list,axis=0)
     
    print("done")
    print("STAGE 3")
    sys.stdout.write("Training SFA2-simple.. ")
    sys.stdout.flush()
    
    #=========STAGE S3: TRAINING OF SFA2======================================
    semantic.train_SFA(sfa2S, SFA1_output)
    
    print("done")
    
    print(semantic.get_d_values(sfa2S))
    
    sys.stdout.write("Training SFA2-episodic on generated scenarios.. ")
    sys.stdout.flush()
    
    semantic.train_SFA(sfa2E, stage2_retrieval_arr)
    
    print("done")
    
    print(semantic.get_d_values(sfa2E))
    
    print("STAGE 4")
    sys.stdout.write("Feeding testing input through entire system.. ")
    sys.stdout.flush()
    
    #=========STAGE 4: TESTING===============================================
    #run SFA chain for both cases
    st4_SFA1_output      = semantic_system.run_SFA(sfa1, testing_sequence)
    SFA2_output_simple   = semantic_system.run_SFA(sfa2S, st4_SFA1_output)
    SFA2_output_episodic = semantic_system.run_SFA(sfa2E, st4_SFA1_output)
    
    print("corr", tools.feature_latent_correlation(SFA2_output_simple, testing_latent))
    
    #tools.preview_input(selection_sequence, slow_features = SFA1_output, retrieved_sequence=SFA2_output_simple, dimensions=4, rate=10, save=False)
    
    print("done")
    sys.stdout.write("Storing output and retrieving several times.. ")
    sys.stdout.flush()
     
    #storage
    if PARAMETERS.constrain_variance:
        SFA2_output_simple /= np.std(SFA2_output_simple)
        SFA2_output_episodic /= np.std(SFA2_output_episodic)

    memtestS.store_sequence(SFA2_output_simple)
    memtestE.store_sequence(SFA2_output_episodic)
     
    #retrieval
    retrieval_length = PARAMETERS.st4['memtest']['retrieval_length']
    retr_simple, jumps_simple, indices_simple, retr_episodic, jumps_episodic, indices_episodic, org_simple, org_episodic = [], [], [], [], [], [], [], []
    for i in range(PARAMETERS.st4['number_of_retrievals']):
        cue_index = random.randint(0,len(SFA2_output_simple)-retrieval_length)
        retr_simple_s, jumps_simple_s, indices_simple_s = memtestS.retrieve_sequence(cue_index, return_jumps=True, return_indices=True)
        retr_episodic_s, jumps_episodic_s, indices_episodic_s = memtestE.retrieve_sequence(cue_index, return_jumps=True, return_indices=True)
        org_simple_s = SFA2_output_simple[cue_index:cue_index+retrieval_length]
        org_episodic_s = SFA2_output_episodic[cue_index:cue_index+retrieval_length]
         
        retr_simple.append(retr_simple_s)
        jumps_simple.append(jumps_simple_s)
        indices_simple.append(indices_simple_s)
        retr_episodic.append(retr_episodic_s)
        jumps_episodic.append(jumps_episodic_s)
        indices_episodic.append(indices_episodic_s)
        org_simple.append(org_simple_s)
        org_episodic.append(org_episodic_s)

    print("done")
     
    d_values = [semantic.get_d_values(s) for s in semantic_system.SFAs]
    
    selection_corr = tools.feature_latent_correlation(SFA1_output, selection_latent, selection_categories)
    testing_corr = tools.feature_latent_correlation(st4_SFA1_output, testing_latent, testing_categories)
    testing_corr_simple = tools.feature_latent_correlation(SFA2_output_simple, testing_latent, testing_categories)
    testing_corr_episodic = tools.feature_latent_correlation(SFA2_output_episodic, testing_latent, testing_categories)
 
    if PARAMETERS.st4['output_to_file']:
        sys.stdout.write("Saving results file.. ")
        sys.stdout.flush()
         
        result_obj = result.Result(PARAMETERS,locals())
        result_obj.save_to_file(PARAMETERS.result_prefix + str(running_ind) + ".p")
        print("done")
 
    #tools.preview_input(selection_sequence, slow_features = SFA1_output, retrieved_sequence=SFA2_output_simple, dimensions=4, rate=10, save=False)
    print("FINISHED")
