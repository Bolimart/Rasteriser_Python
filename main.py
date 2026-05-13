from matplotlib import pyplot
from renderer import *
from time import *


camera = Camera(
    width=400, height=400,
    pos=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),   # looking along +Z
    d=200
)
cube = Cube()
color = [np.array((0,1,0)), np.array((0,1,0)), np.array((0,0,1)), np.array((0,0,1)), np.array((1,1,0)), np.array((1,1,0)), np.array((0,1,1)), np.array((0,1,1)), np.array((1,0,0)), np.array((1,0,0)), np.array((1,0,1)), np.array((1,0,1))]
cube.set_colors(color)
t = time()
objects = [
    Instance(cube, Transform(Point3D(0, 0, 4), Point3D(2, 2, 2), Point3D(45, 45, 90))),
    Instance(cube, Transform(Point3D(0, 0, 10), Point3D(2, 2, 2), Point3D(45, 45, -90))),
    #Instance(cube, Transform(Point3D(0, 0, 2), Point3D(0.5, 0.5, 1), Point3D(0, 25, 15))),
    #Instance(cube, Transform(Point3D(1, 2, 5), Point3D(2, 1, 1), Point3D(0, 0, 45))),
    #Instance(cube, Transform(Point3D(3, -2, 5), Point3D(1, 4, 1), Point3D(15, 0, 0))),
    #Instance(cube, Transform(Point3D(-10, -8, 20), Point3D(10, 10, 10), Point3D(-30, 15, 15)))
]

viewport = Viewport(objects, camera, perspective_projection)
pyplot.imsave(f"img/image_wireframe.png", render_wireframe(viewport, 500, 500))
pyplot.imsave(f"img/image.png", render(viewport, 500, 500))
print(f"Done — saved image.png in {time()-t} seconds")
