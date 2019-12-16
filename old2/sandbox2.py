from core import semantic, system_params, input_params, streamlined, result, semantic_params

PARAMETERS = system_params.SysParamSet()

#PARAMETERS.setsem2(semantic_params.make_layer_series(48,36,40,40,36,10))
PARAMETERS.setsem2(semantic_params.make_layer_series(16,16,20,20,20,16))

PARAMETERS.program_extent = 4

PARAMETERS.st2['object_code'] = input_params.make_object_code('T', 15)
PARAMETERS.st2['sequence'] = [0]

PARAMETERS.st2['movement_type'] = 'timed_border_stroll'
PARAMETERS.st2["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=50, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50
PARAMETERS.st2["snippet_length"] = None
PARAMETERS.st2["interleaved"] = True
PARAMETERS.st2["blank_frame"] = False
PARAMETERS.st2["glue"] = "random"

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['completeness_weight'] = 5
PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

PARAMETERS.st4['object_code'] = input_params.make_object_code('T', 15)
PARAMETERS.st4['sequence'] = [0]
PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50
PARAMETERS.st4["snippet_length"] = None
PARAMETERS.st4["interleaved"] = True
PARAMETERS.st4["blank_frame"] = False
PARAMETERS.st4["glue"] = "random"

# PARAMETERS.preview = True

FILEPATH_SFA1 = "sfaved/poop.sfa"
semantic_system = semantic.SemanticSystem()
sfa1 = semantic_system.load_SFA_module(FILEPATH_SFA1)

res = streamlined.program(PARAMETERS,sfa1)

#tools.compare_inputs((res.stage2_visual,res.comparison_visual),rate=50)

#res.plot_correlation()
#result.show_plots()



#====================================================================================
# PARAMETERS.input_params_default["number_of_snippets"] = 250
# PARAMETERS.input_params_default["snippet_length"] = None
# PARAMETERS.input_params_default["interleaved"] = True
# PARAMETERS.input_params_default["blank_frame"] = False
# PARAMETERS.input_params_default["glue"] = "random"

# PARAMETERS.input_params_default["movement_type"] = 'random_walk'
# PARAMETERS.input_params_default["movement_params"] = dict(dx = 0.05, dt = 0.05, step=1)

# PARAMETERS.input_params_default["movement_type"] = 'lissajous'
# PARAMETERS.input_params_default["movement_params"] = dict( a = 2.3, b = 3.7, deltaX = 3, omega = 1, step = 0.02)
# 
# PARAMETERS.input_params_default["movement_type"] = 'timed_border_stroll'
# PARAMETERS.input_params_default["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=100, border_extent=2.3)

# PARAMETERS.input_params_default["movement_type"] = 'random_rails'
# PARAMETERS.input_params_default["movement_params"] = dict(dx_max=0.05, dt_max=0.1, step=6, border_extent=2.3)

# PARAMETERS.input_params_default["movement_type"] = "random_stroll"
# PARAMETERS.input_params_default["movement_params"] = dict(d2x=0.005, d2t = 0.01, dx_max=0.05, dt_max=0.1)

# sensory_system = sensory.SensorySystem(PARAMETERS.input_params_default, save_input=False)
# testing_sequence, testing_categories, testing_latent = sensory_system.generate()
# 
# tools.preview_input(testing_sequence)
# 
# lat = np.array(testing_latent)
# 
# print("latent shape", np.shape(lat))
# print(PARAMETERS.__dict__)
# print(np.shape(testing_sequence))
# 
# (n, bins, patches) = pyplot.hist(lat[:,2],bins=25, range=(-1,1))
# pyplot.figure()
#  pyplot.hist(np.sin(np.arange(0,2*np.pi,0.01)),bins=25, range=(-1,1))
#  pyplot.figure()
# (counts, xedges, yedges, im) = pyplot.hist2d(lat[:,0],lat[:,1],bins=50, cmin=1,cmap="winter")
# pyplot.colorbar()
# pyplot.figure()
# pyplot.plot(lat[:,0])
# pyplot.plot(lat[:,1])
# print("sin min", np.min(lat[:,2]))
# print("sin max", np.max(lat[:,2]))
# print("xmin", np.min(lat[:,0]))
# print("xmax", np.max(lat[:,0]))
# print("ymin", np.min(lat[:,1]))
# print("ymax", np.max(lat[:,1]))
# print("2d min counts", np.min(counts))
# print("2d max counts", np.max(counts))
# pyplot.show()
