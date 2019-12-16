from core import semantic, system_params, input_params, streamlined, result, semantic_params

PARAMETERS = system_params.SysParamSet()

PARAMETERS.program_extent = 4
PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = False, True
PARAMETERS.filepath_sfa1 = "sfaved/poop.sfa"

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50
# PARAMETERS.st2["snippet_length"] = None
# PARAMETERS.st2["interleaved"] = True
# PARAMETERS.st2["blank_frame"] = False
# PARAMETERS.st2["glue"] = "random"

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['completeness_weight'] = 0
PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50
# PARAMETERS.st4["snippet_length"] = None
# PARAMETERS.st4["interleaved"] = True
# PARAMETERS.st4["blank_frame"] = False
# PARAMETERS.st4["glue"] = "random"

# PARAMETERS.preview = True

res = streamlined.program(PARAMETERS)