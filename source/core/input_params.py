""" 
This module contains the default input parameters -
base parameters as well as movement type specific settings
For more information on the movement types see module :py:mod:`core.trajgen`.
"""

from .tools import listify
from numpy import pi

def make_object_code(letters, sizes=15) :
    """
    :param letters: list or string containing the letters to be shown
    :param sizes: optional, default 15. Size in points of the letters to be generated
    :return: A string that can be evaluated to a dictionary containing generators. Looks like::

           '{
               0: trajgen.presentor(trajgen.makeTextImage('T', outsize=15)),
               1: trajgen.presentor(trajgen.makeTextImage('L', outsize=15))
            }'

        To create a single trajectory with a single alternating image, do this instead::

           '{
               0: trajgen.alternator([trajgen.makeTextImage('T', outsize=15),
                                      trajgen.makeTextImage('L', outsize=15)], 50)
            }'

        where the 50 is the number of frames that each letter is shown before switching.

    """

    return '{'+( ',\n '.join("{} : trajgen.presentor(trajgen.makeTextImageTrimmed('{}', outsize={}))".format(i,l,s) for i,(l,s)
        in enumerate(zip(letters,listify(sizes,len(letters)) )))) + '}'


class BaseParms():
    """
    .. class:: BaseParms

    """

    def __init__(self):
        """
        contains the default input parameters that are not movement-type-specific.
        The parameters ``movement_type`` and ``movement_params`` are added for each movement type individually.
        Some movement types also overwrite ``snippet_length``.

        .. note::
           This class is not meant to be instantiated. It is just there to make documentation of the parameters easier.

        We want to generate ``number_of_snippets`` repeats of the pattern ``sequence``
        where the pattern letters are keys to the dictionary coded for by ``object_code``.
        for instance, with pattern ABC, number_of_snippets = 10, we will generate a 30-long
        series of the form ABCABCABC...ABC
        HOWEVER: if not ``interleaved``, then this looks instead like
        AAA...BBB...CCC...

        .. note::
           In the parameter files, the term *snippet* denotes an individual episode of the generated data.
           However, the parameter ``number_of_snippets`` determines how many episodes of each object
           are generated - e.g. if two objects are used, the number of episodes in the data is 2 :math:`{\cdot}` ``number_of_snippets``.
           How many and which objects there are is determined by the parameters ``sequence`` and
           ``object_code``.

        """
        pass

    #: Pixel dimensions of the images
    frame_shape = (30, 30)
    #: How to interpolate images (before adding noise).
    #: Zero-th order splines are binary images, first order have linear interpolation
    #: for edges, and so forth. This parameter must be an integer in [0,5]
    spline_order = 2
    #: the pixel values for each snippet are in this range,
    #: normalized across the entire snippet, as opposed to each individual image.
    snippet_scale = [0, 1]
    #: Standard deviation of the Gaussian noise that is added to each pixel of each image independently
    input_noise = 0.1
    #: whether to clip to the previous interval after adding noise
    #: If false, pixel values might be outside of the interval determined by ``snippet_scale``
    scale_clip = True
    #: If not None, provide a dictionary like this:
    #: background_params = dict(seed_prob = 0.1, spread_prob = 0.3, spread_scaling = 0.1, constant=False)
    #: Instead of a uniform white background, a method for randomly generating a background is available
    #: With a certain probability, a pixel is selected as a gray seed, from which grey pixels spread with
    #: a certain probability. This background can be set to be constant for all episodes
    background_params = None

    #: list of identity labels that determines in what order different objects are used for input generation
    #: These labels have to be the keys in the dictionary ``object_code``.
    sequence = [0,1]
    #: length of each snippet. Some movement types (timed_border_stroll, random_rails) will overwrite the
    #: value with *None* because according to those movement types, a snippet ends when the object leaves the image.
    snippet_length = 1000
    #: number of snippet to generate for each object in ``sequence``
    number_of_snippets = 1
    #: Whether to present the objects in blocks (*False*) or in an interleaved manner (*True*)
    #: If ``sequence = [0, 1]`` and ``number_of_snippet = 2``, *False* would lead to generating
    #: the series ``0011`` while True would yield ``0101``.
    interleaved = True
    #: A blank fram is inserted between snippets if *True*
    blank_frame = False
    #: How snippets are constrained to be glued together. If this
    #: takes the value 'latent', then snippets pick up latent variables where
    #: they were left off. For starting positions to be randomized, use glue = 'random'.
    glue = 'random'

    #: | A string that contains python code that evaluates to a dict.
    #: | The keys for this dictionary match up with the ``sequence``.
    #: | The values are generators for 2D arrays representing images.
    #:   The generator is most easily built, in the case of a static set of images,
    #:   through either the trajgen.presentor or trajgen.alternator wrappers.
    #:   The images themselves can be loaded, or generated.
    #: | The function :py:func:`make_object_code` in this module is useful for
    #:   generating the string automatically
    #: | In this case, we have two objects, an uppercase *T*, and an uppercase *L*.
    #: | ``object_code = make_object_code('TL')``
    #: | They can also be sent in as a list, as in
    #: | ``object_code = make_object_code(['T', 'L'])``.
    object_code = make_object_code('TL')

############################ PRESET INPUT PARAMETERS ##########################
# A random stroll parameter set. See trajgen.py for details about movement.
_base = dict( BaseParms.__dict__)
priv_keys = [k for k in _base.keys() if k[:2] == "__"]
for pk in priv_keys:
    _ = _base.pop(pk)

_stroll = dict(
    # Here is where you should look at trajgen.py for information.
    movement_params = dict(d2x=0.005, d2t = 0.009, dx_max=0.05, dt_max=0.1, step=1),
    movement_type = 'random_stroll'
)

stroll = None  #: Default parameters for ``random_stroll``

_walk = dict(
    movement_params = dict(dx = 0.05, dt = 0.05, step=1),
    movement_type = 'random_walk'
)

walk = None  #: Default parameters for ``random_walk``

_gausswalk = dict(
    movement_params = dict(dx = 0.05, dt = 0.05, step=5),
    movement_type = 'gaussian_walk'
)

gausswalk = None  #: Default parameters for ``gaussian_walk``

# A Lissajous input parameter set. See trajgen.py for details about movement. 
_lissa = dict(            # (copy stroll dict first, then overwrite)
     movement_params = dict( a = 6,   # horizontal frequency
                b = 2*pi,             # vertical frequency
                deltaX = 3,           # Phase offset in x-coordinate
                omega = 7,            # rotations per cycle
                step = 0.02),         # Phase update per frame (2pi cycle)
    movement_type = 'lissajous'
)

lissa = None  #: Default parameters for ``lissajous``

_catsndogs = dict(
    movement_params = dict(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=2, nframes_border=50, border_extent=2.3),
    movement_type = 'timed_border_stroll',
    snippet_length = None
)

catsndogs = None  #: Default parameters for ``timed_border_stroll``

_rails = dict(
    movement_params = dict(dx_max=0.05, dt_max=0.1, step=1, border_extent=2.3),
    movement_type = 'random_rails',
    snippet_length = None
)

rails = None  #: Default parameters for ``random_rails``

_still = dict(
    movement_params=dict(),
    movement_type='stillframe',
    snippet_length = 2
)

still = None  #: Default parameters for ``stillframe``

_rand = dict(
    movement_params=dict(),
    movement_type='uniform',
    snippet_length = 1000
)

rand = None  #: Default parameters for ``uniform``

# The above private dictionaries are only partial; they have only the movement
# specific parameters. To complete this class so it works like it used to,
# with the full parameter set being accessible, we have to start with base and
# override specific parameters.
for k,v in list(locals().items()):    # copy into list here, otherwise we get dict size change during iteration
    if k.startswith('_') and isinstance(v,dict):
        exec(k[1:]+' = dict(_base, **'+k+')')