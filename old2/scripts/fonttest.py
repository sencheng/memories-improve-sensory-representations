import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageChops
from matplotlib import pyplot

text = 'L'
fontfile = '../fonts/Lato-Black.ttf'
result = []

def trim(arr):
    B = np.argwhere(arr)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    atrim = arr[ystart:ystop, xstart:xstop]
    return atrim

for outsize in range(10, 100, 10):

    img = Image.fromarray(np.zeros((2*outsize, 2*outsize)))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(fontfile, outsize)

    tsize = font.getsize(text)

    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((outsize/2, outsize/2), text, 1, font=font)
    #draw.text(((1.1*outsize - tsize[0]) / 2, 1.1*outsize / 2 - tsize[1]), text, 1, font=font)
    #draw.text((0,0), text, 1, font=font)
    arr = np.asarray(img)
    atrim = trim(arr)
    result.append(atrim)
    result.append(arr)


#np.save("fontimg.npy", result)

tot = len(result)
for i in range(tot):
    pyplot.subplot(1,tot,i+1)
    pyplot.imshow(result[i], interpolation='none')
pyplot.show()