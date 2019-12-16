from core import semantic, sensory, system_params

import profile

incsfa_parms1 = [
        ('inc.linear', {
            'dim_in': 900,
            'dim_out': 16
        })
    ]

PARAMETERS = system_params.SysParamSet()

PARAMETERS.st1['number_of_snippets'] = 50
PARAMETERS.st1['input_noise'] = 0.2

sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
print("Generating input")
training_sequence, training_categories, training_latent = sensory_system.generate(fetch_indices=False, **PARAMETERS.st1)

print("Building SFA")
incsfa1 = semantic.build_module(incsfa_parms1)
print("Training incSFA1")
profile.run("semantic.train_SFA(incsfa1, training_sequence)", "../results/incprof/incprof.stat")


# print("Executing incSFA1")
# inc_seq1 = semantic.exec_SFA(incsfa1, training_sequence)
# print("Executing incSFA2")
# inc_seq2 = semantic.exec_SFA(incsfa2, training_sequence)

# print("Executing bSFA1")
# b_seq1 = semantic.exec_SFA(bsfa1, training_sequence)
# print("Executing bSFA2")
# b_seq2 = semantic.exec_SFA(bsfa2, training_sequence)