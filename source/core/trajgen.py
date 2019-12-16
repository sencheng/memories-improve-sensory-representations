# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.ndimage.interpolation as interp
from random import random
from itertools import islice
from PIL import Image, ImageFont, ImageDraw
import os

""" These three methods make the rest of the code a lot cleaner. """ 
def _clamp(x, b): return min(b, max(-b, x ))
def _rand():  return 2*random()-1
def _randN(ds) : return np.random.normal(0, ds, 1)[0]
def _randU(ds) : return np.random.uniform(-ds, ds, 1)[0]

def trim(arr):
    """
    Crops an image, removes white border

    :param arr: input image
    :return: cropped image
    """
    B = np.argwhere(arr)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    atrim = arr[ystart:ystop, xstart:xstop]
    return atrim

def makeTextImageTrimmed(text, outsize=25, fontfile=None) :
    """
    Generates a cropped image with text

    :param text: String to print on image
    :param outsize: Font size in points.
    :param fontfile: TTF file or similar. If none, default font *Lato-Black.ttf* is used.
    :return: Image containing text
    """
    img = Image.fromarray(np.zeros((2 * outsize, 2 * outsize)))
    draw = ImageDraw.Draw(img)

    if not fontfile:
        this_file = os.path.abspath(__file__)
        this_path = this_file[:-len("/trajgen.py")]
        source_path = os.path.abspath(os.path.join(this_path, os.pardir))
        fontfile = os.path.join(source_path, "fonts", "Lato-Black.ttf")
        # fontfile = 'fonts/Lato-Black.ttf'
    font = ImageFont.truetype(fontfile, outsize)

    #tsize = font.getsize(text)

    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((outsize / 2, outsize / 2), text, 1, font=font)
    # draw.text(((1.1*outsize - tsize[0]) / 2, 1.1*outsize / 2 - tsize[1]), text, 1, font=font)
    # draw.text((0,0), text, 1, font=font)
    arr = np.asarray(img)
    atrim = trim(arr)
    return atrim

def makeTextImage(text, outsize=25, fontfile=None) :
    """
    Generates an image with text. The crop method used here is not as sophisticated, so result
    might have white border or be cur off.

    :param text: String to print on image
    :param outsize: Font size in points.
    :param fontfile: TTF file or similar. If none, default font *Lato-Black.ttf* is used.
    :return: Image containing text
    """
    img = Image.fromarray(np.zeros((outsize,outsize)))
    draw = ImageDraw.Draw(img)
    
    if not fontfile:
        this_file = os.path.abspath(__file__)
        this_path = this_file[:-len("/trajgen.py")]
        source_path = os.path.abspath(os.path.join(this_path, os.pardir))
        fontfile = os.path.join(source_path, "fonts", "Lato-Black.ttf")
        # fontfile = 'fonts/Lato-Black.ttf'
    font = ImageFont.truetype(fontfile, outsize)
        
    tsize = font.getsize(text)

    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text(((outsize-tsize[0])/2, outsize/2-tsize[1]), text, 1,font=font)
    return np.asarray(img)
    

def alternator(images, nframes_for_each):
    """
    Construct a generator that presents alternating images. Image identity will be switched
    after *nframes_for_each* frames. Helps building the :py:func:`trajGenMaker`.

    :param images: iterable containing images to present
    :param nframes_for_each: number of frames per image
    :return: Generator
    """
    while(True):
        for img in images:
            for i in range(nframes_for_each):      #@UnusedVariable
                yield img

def presentor(image):
    """
    Construct a generator that will present the same image all the time.
    Helps building the :py:func:`trajGenMaker`.

    :param image: Image to present
    :return: Generator
    """
    while(True):
        yield image

def makeTransform(theta, dx, dy,scale=1):
    """
    Make a 3x3 homogeneous matrix that represents a 2-dimensional affine transform
    of the form  [R t], where R is a rotation by theta, and t is a translation by (dx,dy)
    The optional scale parameter scales. This is mostly a convenience method.

    :param theta: rotation angle
    :param dx: translation in x-direction
    :param dy: translation in y-direction
    :param scale: Scaling factor
    :return: Affine transformation matrix
    """
    y, x =  math.sin(theta) * scale, math.cos(theta) * scale
    return np.matrix([[x, -y, dx], [y, x, dy], [0,0,1]])
    
    
def trajGenMaker(image_iterable, dim, updater, spline_order=0) :
    """
    Creates a generator that generates image frames that move inside a larger box.
    and the box is of size dim. The image then rotates/scales/shears, depending on
    the updater generator/iterable/array-like.

    Note that (0,0) is the center of the image, because that makes math easier, and allows
    default centering regardless of size parameters. The range [-1, 1]x[-1,1] corresponds to
    the area in which the image is (roughly) entirely within the border.

    :param image_iterable: Iterable yielding images to put in matrix format
    :param dim: pixel dimensions of the resulting images
    :param updater: generator/iterable/array-like yielding affine transformation matrices.
                    The generator can be created using one of the mover functions in this module
    :param spline_order: Interpolation method, 0 means binary images,
                         1 is linear interpolation for edges, and so forth. This parameter must be an integer in [0,5]
    :return: Generator yielding image frames

    """

    first = next(image_iterable)
    sh = np.shape(first)
    
    def generator() :
        while True:
            transform, latent_vars = next(updater)
            image = next(image_iterable)
            
            for i in [0,1]:
                transform[i,2] *= (dim[i] - sh[i]*math.sqrt(np.linalg.det(transform)))/2
                        # We want to rescale the translational component
                        # so that it matches the [-1,1] parameter space (so that it can 
                        # be independent of image length)
            t =  makeTransform(0, sh[0]/2, sh[1]/2) * transform.I * makeTransform(0,-dim[0]/2,-dim[1]/2) 
                # fancy matrix multiplication to get the right transformation (conjugate by translations)       

            yield interp.affine_transform(image, t[:2,:2], offset=[t[0,2],t[1,2]], 
                                          output_shape=dim, prefilter=True, order=spline_order), latent_vars
            
    generator.shape = dim # just for convenience, to know what a generator will give you back.
    generator.size = np.prod(dim)
    
    return generator

def sample(x_range=None, y_range=None, t_range=None, x_step=0.01, y_step=0.01, t_step=45):
    """
    Iterate through xy-positions and angles

    :param x_range: Iterable yielding values (or a single scalar) that evaluate to the desired x-coordinates
                    by multiplying with *x-step*. E.g. ``x_range=range(-100,101)`` and ``x_step=0.01`` would nicely fit together to cover the entire image *[-1,1]*.
    :param y_range: Same for y.
    :param t_range: Iterable yielding values (or a single scalar) that evaluate to the desired angles in pi (so 1 means pi)
                    by multiplying with *t_step/180*. E.g. ``t_range=range(-3, 4)`` and ``t_step=45`` would nicely fit together to cover one rotation *(-2,2]*.
    :param x_step: step size in x direction
    :param y_step: step size in y direction
    :param t_step: angular increment in degrees
    :return: Generator yielding transformation matrices
    """
    circle_step = t_step/360.
    if x_range is None:
        x_range = range(int(-1 / x_step), int(1 / x_step) + 1)
    elif not hasattr(x_range, '__iter__'):
        x_range = [x_range]
    if y_range is None:
        y_range = range(int(-1 / y_step), int(1 / y_step) + 1)
    elif not hasattr(y_range, '__iter__'):
        y_range = [y_range]
    if t_range is None:
        t_range = range(int(-0.5 / circle_step) + 1, int(0.5 / circle_step) + 1)
    elif not hasattr(t_range, '__iter__'):
        t_range = [t_range]

    def gen(latent_start=None):

        for x in x_range:
            x_coord = x*x_step
            for y in y_range:
                y_coord = y*y_step
                for t in t_range:
                    t_pi = np.pi*t*t_step/180.
                    yield makeTransform(t_pi, x_coord, y_coord), [x_coord, y_coord, math.cos(t_pi), math.sin(t_pi)]

    return gen

def stillframe():
    """
    At initialization, choose a random location and orientation and keep that constant
    to generate a still image.

    :return: Generator yielding always the same transformation matrix.
    """
    
    def gen(latent_start=None):
        x, y, t = _rand(), _rand(), _rand()*np.pi
        while True:
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t)]
            
    return gen
    
def uniform():
    """
    Location and orientation are randomly drawn from a uniform distribution for every frame.

    :return: Generator yielding transformation matrices
    """
    def gen(latent_start=None):
        while True:
            x, y, t = _rand(), _rand(), _rand()*np.pi
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t)]

    return gen

def copy_traj(latent, ranges):
    """
    Move and rotate the object according to the latent variable sequence supplied by user

    :param latent: latent variable sequence
    :param ranges: iterator over the ranges of snippets to generate
    :return: Generator yielding transformation matrices
    """
    def gen(latent_start=None):
        idx = 0
        lat = np.array(latent)[list(next(ranges))]

        while idx<len(lat):
            [x,y,cos_t, sin_t] = lat[idx][:4]
            t = np.arctan2(sin_t,cos_t)
            yield makeTransform(t,x,y), lat[idx]
            idx += 1

    return gen

def lissajous(a, b, deltaX, omega, deltaY=0, start_angle=0, step=0.01):
    """
    Lissajous parametric curve:

    |        :math:`{x(t) = sin(a*t + deltaX)}`
    |        :math:`{y(t) = sin(b*t + deltaY)}`

    Rotation is included as follows:

    :math:`{\phi(t) = start\_angle + omega*t}`


    :param a:
    :param b:
    :param deltaX:
    :param omega:
    :param deltaY:
    :param start_angle:
    :param step: Increment of the t parameter. Basically speed.
    :return: Generator yielding transformation matrices
    """
    
    def gen(latent_start=None):
        time = 0
    
        if latent_start is None:
            time = _rand()*517 # this is large enough to ensure that phase is overwhelmingly 
                # to be scrambled, especially because it's not a multiple of pi
        else:
            time = latent_start[4]            
            
        while True:
            time += step
            t,x,y = start_angle + omega*time, math.sin(time*a + deltaX), math.sin(time*b + deltaY)
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t), time]
    
    return gen
    
def random_walk(dx, dt, step=1):
    """
    Random walk (brownian motion), drawing from a uniform distribution

    :param dx: Half width of the uniform zero-mean distribution to draw difference values for both x-
               and y-coordinate from
    :param dt: Same for rotation (in rad)
    :param step: Scale parameter for all latents. Basically speed.
    :return: Generator yielding transformation matrices
    """
    
    def gen(latent_start=None):
        x, y, t = 0,0,0
        
        if latent_start is None:
            x, y, t = _rand(), _rand(), _rand()*np.pi
            if dt == 0:
                t = 0
        else:
            x, y, cost, sint = latent_start
            t = math.atan2(sint, cost)
        
        while True:
            x = _clamp(x + dx*_randU(step), 1)
            y = _clamp(y + dx*_randU(step), 1)
            t = t + dt*_randU(step)
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t)]
            
    return gen
      

def gaussian_walk(dx, dt, step=1):
    """
    Random walk (brownian motion), drawing from a Gaussian distribution

    :param dx: Standard deviation of the zero-mean Gaussian distribution to draw difference values for both x-
            and y-coordinate from
    :param dt: Same for rotation (in rad)
    :param step: Scale parameter for all latents. Basically speed.
    :return: Generator yielding transformation matrices
    """
    """Similar to the random walk, but change in position and rotation are drawn from a normal distribution."""
    
    def gen(latent_start=None):
        x, y, t = 0,0,0
        
        if latent_start is None:
            x, y, t = _rand(), _rand(), _rand()*np.pi
            if dt == 0:
                t = 0
        else:
            x, y, cost, sint = latent_start
            t = math.atan2(sint, cost)
        
        while True:
            x = _clamp(x + dx*_randN(step), 1)  #Adjust coefficient of step size for coverage
            y = _clamp(y + dx*_randN(step), 1)
            t = t + dt*_randN(step)
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t)]
            
    return gen
        
      
def random_stroll(d2x=0.005, d2t=0.009, dx_max=0.1, dt_max=0.2, step=1):
    """
    Integral of a random walk. That means, the change in translation/rotation speed
    is drawn from a Gaussian distribution.

    :param d2x: Standard deviation of the zero-mean Gaussian distribution to draw speed difference values for both x-
                and y-coordinate from
    :param d2t: Same for rotation (in rad/dt)
    :param dx_max: Maximum translation speed in both x- and y-direction
    :param dt_max: Maximum rotation speed
    :param step: Scale parameter for all latents. Basically acceleration.
    :return: Generator yielding transformation matrices
    """
    
    def gen(latent_start=None):
        x,y,t = 0,0,0
        dx,dy,dt = 0,0,0
        
        if latent_start is None:
            x, y, t = _rand(), _rand(), _rand()*np.pi
            dx, dy, dt = _rand()*dx_max, _rand()*dx_max, _rand()*dt_max
        else:
            x, y, cost, sint, dt, dx, dy = latent_start
            t = math.atan2(sint, cost)
            
        while True:
            dx = _clamp(dx+ d2x*_randN(step), dx_max)
            dy = _clamp(dy+ d2x*_randN(step), dx_max)
            dt = _clamp(dt+ d2t*_randN(step), dt_max)
            x += dx*step
            y += dy*step
            t += dt*step
            
            if x < -1 or x > 1:
                x = _clamp(x, 1)
                dx = -x*abs(dx)
            if y < -1 or y > 1:
                y = _clamp(y, 1)
                dy = -y*abs(dy)
    
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t), dt, dx, dy]
            
    return gen

def random_rails(dx_max=0.05, dt_max=0.1, step=1, border_extent=1.6):
    """
    Generate a trajectory that just shoot across the screen with both constant linear
    and angular velocities. These velocities are randomly drawn from a zero-mean uniform distribution.
    The object starts off screen, moves through the screen and the
    generator terminates when the object reaches *border_extent* in either coordinate.

    :param dx_max: Half width of the distribution to draw velocity from
    :param dt_max: Half width of the distribution to draw angular velocity from
    :param step: Scale parameter for all latents. Basically speed.
    :param border_extent: Terminating condition for the generator. The image is
                          considered off-screen if the absolute value of one of the coordinates reaches *border_extent*.
    :return: Generator yielding transformation matrices
    """

    def gen(latent_start=None) :
        x,y,t = 0,0,0
        dx,dy,dt = 0,0,0
        
        if latent_start is None:
            # Instead of selecting a random position on the screen, we select a random position
            # on the border. But to make the distribution realistic, we randomize a direction uniformly
            # and then project it onto a square border.

            place = _rand()*np.pi                   # the angle from the center
            y = np.clip(math.tan(place)*np.sign(np.cos(place)), -1, 1) * border_extent      # projection of y coordinate
            x = np.clip(1/math.tan(place)*np.sign(np.sin(place)), -1, 1) * border_extent    # projection of x coordinate
            
            # Also, we want a uniform covering of pixels if possible. So from the starting position,
            # uniformly randomize an angle within the range that will leave it completely on the screen
            # for at least one frame. However, squares are difficult to work with, and hurt if they fall
            # on you, so we will instead only care about randomizing uniformly on an inscribed circle of the
            # screen. The corners will still get pixels, but we won't promise them as much (sort of like women).
            half_spread = math.atan(1/border_extent)
            direction = place + half_spread*_rand()
            speed = dx_max*(random()+1)/2
            dx = (-math.cos(direction))*speed
            dy = (-math.sin(direction))*speed

            t, dt = _rand()*math.pi, _rand()*dt_max
        else:
            x, y, cost, sint, dt, dx, dy = latent_start
            t = math.atan2(sint, cost)
            
        while True:
            x += dx*step
            y += dy*step
            t += dt*step
            
            if abs(x) > border_extent or abs(y) > border_extent :     # The object disappears
                break
            
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t), dt, dx, dy]
        
    return gen
    
def timed_border_stroll(d2x=0.005, d2t=0.009, dx_max=0.05, dt_max=0.1, step=1, nframes_border=1, border_extent=1.6):
    """
    Random stroll, except objects start outside the boundary and are shot towards the center
    (with a small random component) before performing the random stroll. After *nframes_border*
    frames have been generated, the bouncy borders disappear and the object can diffuse off screen.
    When it reaches *border_extent*, the generator terminates.

    :param d2x: Standard deviation of the zero-mean Gaussian distribution to draw speed difference values for both x-
                and y-coordinate from
    :param d2t: Same for rotation (in rad/dt)
    :param dx_max: Maximum translation speed in both x- and y-direction
    :param dt_max: Maximum rotation speed
    :param step: Scale parameter for all latents. Basically acceleration.
    :param nframes_border: Number of frames to remain in random stroll.
    :param border_extent: Terminating condition for the generator. The image is
                          considered off-screen if the absolute value of one of the coordinates reaches *border_extent*.
    :return: Generator yielding transformation matrices
    """
    """ """
    
    def gen(latent_start=None):
        x,y,t = 0,0,0
        dx,dy,dt = 0,0,0
        countdown = nframes_border
        
        if latent_start is None:
            # Instead of selecting a random position on the screen, we select a random position
            # on the border. But to make the distribution realistic, we randomize a direction uniformly
            # and then project it onto a square border.
        
            place = _rand()*np.pi                   # the angle from the center
            y = np.clip(math.tan(place)*np.sign(np.cos(place)), -1, 1) * border_extent      # projection of y coordinate
            x = np.clip(1/math.tan(place)*np.sign(np.sin(place)), -1, 1) * border_extent    # projection of x coordinate
            dx = (_rand()/3-2*math.cos(place))*dx_max   # Also, we want velocity to start towards the middle,
            dy = (_rand()/3-2*math.sin(place))*dx_max   # more or less...

            t, dt = _rand()*math.pi, _rand()*dt_max
            
        else:
            x, y, cost, sint, dt, dx, dy = latent_start
            t = math.atan2(sint, cost)

        inside = abs(x) < 1 and abs(y) < 1   # Do we start inside the arena? If not, don't clamp
                        # positions at first. Instead let it get inside before enforcing edges.          

        while True:
            if inside:
                dx = _clamp(dx+ d2x*_randN(step), dx_max)
                dy = _clamp(dy+ d2x*_randN(step), dx_max)
            dt = _clamp(dt+ d2t*_randN(step), dt_max)
            x += dx*step
            y += dy*step
            t += dt*step
            
            inside = inside or (abs(x) < 1 and abs(y) < 1)
            
            if countdown > 0 :
                if inside:
                    if abs(x) > 1:
                        x = _clamp(x, 1)
                        dx = -x*abs(dx)
                    if abs(y) > 1:
                        y = _clamp(y, 1)
                        dy = -y*abs(dy)
            if abs(x) > border_extent or abs(y) > border_extent :     # The object disappears
                break      # after reaching coordinate +/- <border_extent>. Breaking stops iteration of generator.
                
            countdown -= 1
    
            yield makeTransform(t,x,y), [x,y, math.cos(t), math.sin(t), dt, dx, dy]
            
    return gen

# ==================================================
# FUNCTIONS NEEDED FOR RANDOM BACKGROUND CALCULATION
def surround_list(l, dims, container=None):
    env = [(l[0]+d0, l[1]+d1) for d0 in range(-1,2) for d1 in range(-1,2)]
    sur = [el for el in env if el[0] < dims[0] and el[1] < dims[1]]
    if container is None:
        return sur
    val = [el for el in sur if el in container]
    return val

def choose(l, count=1):
    if count == 1:
        return l[np.random.randint(len(l))]
    ret = []
    inds = np.random.choice(len(l), count, replace=False)
    for i in inds:
        ret.append(l[i])
    return ret

def spread(p):
    return 1 if np.random.random() < p else 0

def bgr(dims, seed_prob, spread_prob, spread_scaling):
    seed_count = int((dims[0]*dims[1])*seed_prob)

    im = np.zeros(dims)

    coords = [(x,y) for x in range(dims[0]) for y in range(dims[1])]  # an element is removed if seed-placement was run on that coord, irrespective of the outcome. No elements can be added.
    seeds = []                                                        # an element is added when a seed is placed. An element is removed when all neighbors were removed from coords.

    seed_locations = choose(coords, seed_count)
    for loc in seed_locations:
        im[loc] = np.random.rand()
        coords.remove(loc)
        seeds.append(loc)

    while len(seeds) > 0:
        here = choose(seeds)
        surround = surround_list(here, dims, coords)
        if len(surround) == 0:
            seeds.remove(here)
            continue
        new = choose(surround)
        prob = spread_prob + spread_scaling*len(surround_list(new, dims, seeds))
        im[new] = im[here]*spread(prob)

        coords.remove(new)
        seeds.append(new)

    return im
# ==================================================

def movie(nsteps, dims, trajGen, noise_std=0, background_params = None, scale_clip=True, flatten=True, backdef=None):
    """
    Uses the generator provided by :py:func:`trajGenMaker` to make a movie array of desired length.
    The frames provided by the generator are processed according to settings.
    Noise (with clipping) and background can be added. Arrays can be flattened.

    :param nsteps: length of the movie, snippet to generate
    :param dims: pixel dimensions of the images the generator provides
    :param trajGen: generator of the images
    :param noise_std: standard deviation of the Gaussian noise to add to every pixel independently
    :param background_params: None or Dictionary containing background parameters.
    :param scale_clip: Whether to clip the images to the original value range after adding noise
    :param flatten: Whether to flatten the images (True) or keep the matrix shape (False)
    :param backdef: None or Background image, if the background is not supposed to be generated but is predefined
    :return: tuple (movie, latent_vars[, backdef]). *movie* is the generated snippet. Latent_vars is the corresponding
             latent variable sequence. Backdef is only returned if a background was added and contains the raw background.
    """
    frames =  []
    latent_vars = []

    if background_params is not None:
        if backdef is None or not background_params['constant']:
            back = bgr(dims, background_params['seed_prob'], background_params['spread_prob'], background_params['spread_scaling'])
            backdef = back if background_params['constant'] else None
        else:
            back = backdef
        
    # Note that this will also not slice if nsteps is None.
    iterator = islice(trajGen(),nsteps) if nsteps is not None and nsteps > 0 else trajGen()

    clip_min = 10e8
    clip_max = -10e8
    
    for x, lat in iterator:
        frame = x.flatten() if flatten else x
        if background_params is not None:
            thres = (frame < 0.6)
            frame = np.logical_not(thres)*frame + back.flatten()*thres if flatten else np.logical_not(thres)*frame + back*thres

        if scale_clip:
            clip_min = min(clip_min, int(np.min(frame)))
            clip_max = max(clip_max, int(np.max(frame)))

        if noise_std > 0:
            frame += np.random.normal(0, noise_std,np.shape(frame))

        frames.append(frame)
        latent_vars.append(lat)

    if scale_clip:
        ret_frames = np.clip(np.array(frames), clip_min, clip_max)
    else:
        ret_frames = np.array(frames)

    return ret_frames, latent_vars, backdef
    #return list(islice(trajGen(),nsteps))

