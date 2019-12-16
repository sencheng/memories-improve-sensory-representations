"""
This module contains the :py:class:`SensorySystem` class, its objects are used to generate input episodes

"""

import numpy as np
import copy
import itertools
from . import trajgen


class SensorySystem(object):
    """
    .. class:: SensorySystem

    """
    def __init__(self, default_params, save_input=False):
        """
        Create sensory system with the given default parameter set.

        :param default_params: dictionary with default parameters for input generation.
                               For details see :py:mod:`core.input_params`.
        :param save_input: Whether or not to save generated input episodes in the instance.
                           Setting to True makes usage of :py:func:`recall` possible.
        """
        self.params_default = default_params
        self.save_input = save_input # whether or not to actually save input (can be slow and big)
        self.saved = {}  # a mapping from SFA to a list of the history: ([frames], [latents])
        self.background = None  # required if background is set to be constant through several movie calls
        
    def _generate_input(self, numbers=None, gen=True, split=False, fetch_indices=False, **override):
        """
        This is a private method that does either generation or recall. They are merged into
        a single method, because the stitching process is similar in both cases.
        
        By default, this returns a triple ( [frames], [object labels], [latent variables] ), where each
            is a numpy arrays of length equal to the total number of frames.
        
        :param gen: Generate new episodes, or look up parts of old ones
        :param numbers: which snippets to select/generate
        """
        params = copy.copy(self.params_default)
        # Temporarily override any parameters with the override.
        for k,v in override.items():
            if k in params:
                params[k] = v
        
        # This could come in handy.        
        BLANK_FRAME = np.zeros(params['frame_shape']).flatten()
                
        # Now we're ready to actually generate input. Much less parameter shuffling, see?
        # First create our blank arrays and shit
        current = [0] # damn I hate python 2.7; this shouldn't need to be a list; it only needs to be non-local. 
        labels, film, latent = [], [], []

        if fetch_indices:
            snipranges = []
        
        mover = getattr(trajgen, params['movement_type'])(**params['movement_params'])
        objects = eval(params['object_code'])
        
        # The next part is a bit trickier to follow but I have faith in you, young documentation
        # reader. First, recall and generation both involve building a single input, like this:
        def extend_input(movie, latent_vars, label) :
            # Normalize input to params['snippet_range'], which is an array of [min, max].
            mmin, mmax, (wmin, wmax) = movie.min(), movie.max(), params['snippet_scale']
            factor = (wmax-wmin)/(mmax-mmin) if not np.isclose(mmax,mmin) else 1
            movie *= factor
            movie += wmin - mmin*factor

            film.append(movie)
            latent.append(latent_vars)
            labels.append([label]*len(movie))

            if fetch_indices:
                length = np.shape(movie)[0]
                snipranges.append(range(current[0],current[0]+length))
                current[0] += length
            
            if params['blank_frame'] :
                film.append([BLANK_FRAME])
                labels.append([label])
                latent.append([None]) # The blank frames have no latent parameters. 
                    # People doing statistics deserve to have errors if they try
                    # to penalize a learning system for getting a blank screen wrong.
        
        # I dedicate this method to our beloved snippet.
        def snip(label, i):
            state = None
            
            if params['glue'] is 'latent' and len(latent) > 0:
                state = latent[-2 if params['blank_frame'] else -1][-1]

            # Generate new animations
            movie, latent_vars, self.background = trajgen.movie(params['snippet_length'], params['frame_shape'],
                       trajgen.trajGenMaker(objects[label], params['frame_shape'], 
                                            mover(state), spline_order=params['spline_order'] ),
                            noise_std=params['input_noise'], background_params=params['background_params'], scale_clip=params['scale_clip'], backdef=self.background)
            
            # That was a complicated line of code, so I will explain; here are the method signatures:
            #   movie(length, frame generator = trajGenMaker(image, shape, matrixGenerator) )
            # ** Note that params[movement] refers not to a generator function, but instead a function that
            # produces one, so we have to call it first. This is necessary for random starting positions
            # to be re-randomized each snippet, as opposed to when the trajectory class is decided.
                       
            if self.save_input:
                if not label in self.saved:
                    self.saved[label] = []
                           
                self.saved[label].append( (movie, latent_vars) )
            extend_input(movie, latent_vars, label)

        
        # Snapping compliments snipping. 
        def snap(label, i):
            # Recall saved animations instead of generating new ones. 
            movie, lvars = self.saved[label][i]
            extend_input(movie, lvars, label)
            
            
        fun = snip if gen else snap
            
        # Finally, we can use trajgen to generate input according to the 'sequence' variable. Depending on
        # whether they are supposed to be all together. This might look unnecessarily complex, but it is
        # really just the same two for-loops presented inside-out depending on interleaving.
        if numbers is None:
            numbers = range(params['number_of_snippets'])

        if params['interleaved']:
            for i in numbers:
                for letter in params['sequence']:
                    fun(letter, i)
        else:
            for letter in params['sequence']:
                for i in numbers:
                    fun(letter, i)
                    
        if not split:
            # Concatenations default to the zeroth dimension, so we're okay.
            # However, latent might not be all uniform dimensions so we'll have to 
            # a bit of number shuffling first.
            film, labels, latent = np.concatenate(film), np.concatenate(labels), list(itertools.chain.from_iterable(latent))
                        
        
        # Append snipranges if we need to return indices also.
        return (film, labels, latent) + ((snipranges,) if fetch_indices else ())

    def generate(self, **override):
        """
        This is the public interface for generating input

        :param override: keyword arguments to override default parameters with (for this call of *generate* only)
                         best passed as a dictionary using \*\* (double asterisk)
        :return: tuple *(seq, cat, lat)*, where *seq* are the generated epsiodes as ndarray, *cat* are the corresponding
                 object identity labels and *lat* the corresponding latent variables
        """
        return self._generate_input( gen=True, **override )
        
    def recall(self, numbers, **override):
        """
        Instead of generating new input, old input is recalled in the order defined by ``numbers``.
        Only works if attribute ``save_input=True`` when input was generated.

        :param numbers: Indices of the snippets to return. E.g. when 10 snippets were generated, ``numbers`` is an
                        array containing numbers between 0 and 9 in whatever order.
        :param override: Does not do anything here, I don't know why this is a parameter here
        :return: tuple *(seq, cat, lat)*, where *seq* are the recalled episodes as ndarray, *cat* are the corresponding
                 object identity labels and *lat* the corresponding latent variables
        """
        return self._generate_input( numbers=numbers, gen=False, **override )


