from matplotlib import pyplot
from renderer import *

pyplot.imsave("image.png", render(None, 40, 40))