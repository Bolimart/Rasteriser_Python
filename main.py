from matplotlib import pyplot
from renderer import *
from time import *

t = time()
camera = Camera(
    width=400, height=400,
    pos=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),   # looking along +Z
    d=1
)
objects = [

]
viewport = Viewport(objects, camera, perspective_projection)
pyplot.imsave(f"img/image.png", render(viewport, 400, 400))
print(f"Done — saved image.png in {time()-t} seconds")