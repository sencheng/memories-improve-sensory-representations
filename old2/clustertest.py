from core import semantic, system_params, input_params, streamlined, result, semantic_params

PARAMETERS = system_params.SysParamSet()

PARAMETERS.filepath_sfa1 = "../results/clustertest.sfa"

PARAMETERS.load_sfa1, PARAMETERS.save_sfa1 = False, False
PARAMETERS.program_extent = 4
PARAMETERS.which = 'SE'

PARAMETERS.same_input_for_all = False

PARAMETERS.input_params_default['frame_shape'] = (15,15)

PARAMETERS.sem_params1 = [('layer.square', {
                            'bo':               15,
                            'rec_field_ch':     9,
                            'spacing':          2,
                            'in_channel_dim':   1,
                            'out_sfa_dim1':     48,
                            'out_sfa_dim2':     32
                        }),
                        ('layer.square', {
                            # <bo>=  (bo-rec_field_ch)/spacing+1 := 4
                            'rec_field_ch':     3,
                            'spacing':          1,
                            'out_sfa_dim1':     8,
                            'out_sfa_dim2':     32
                        }),
                        ('single', {
                            'dim_in':      128,
                            'dim_mid':     48,
                            'dim_out':     16
                        })]

PARAMETERS.st2['movement_type'] = 'random_rails'
PARAMETERS.st2['movement_params'] = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3)
PARAMETERS.st2["number_of_snippets"] = 50

PARAMETERS.st2['memory']['category_weight'] = 0
PARAMETERS.st2['memory']['completeness_weight'] = 0
PARAMETERS.st2['memory']['retrieval_noise'] = 0.2
PARAMETERS.st2['memory']['depress_params'] = dict(cost=5, recovery_time_constant=10, activation_function='lambda X : X')

# PARAMETERS.st4['movement_type'] = 'timed_border_stroll'
PARAMETERS.st4["movement_params"] = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=400, border_extent=2.3)
PARAMETERS.st4["number_of_snippets"] = 50

res = streamlined.program(PARAMETERS)