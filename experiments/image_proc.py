from skimage.filters import rank
from skimage.morphology import square
from skimage.transform import resize

"""
Enhance and randomize with given parameters.

This function will provide the functionalities of image enhancement and  image
randomization. We want an image to provide more details for CNN, and provide a
degree of random to make sure we have a desirable learning rate.

Parameters
----------
im : array
size: int
random: float

Returns
-------
array
    Image data after enhancement and randomization.
"""
def randomizeEnhanced(im, size, random):
  im = resize(im, (size, size))
  im = rgb2gray(im)
  noise = random*(np.random.random_sample(size=(size, size)) - 0.5)
  im_le = rank.equalize(im, selem=square(80)) - noise
  return im_le