from matplotlib import pyplot
from renderer import *
from time import *

crate_atlas = Atlas(pyplot.imread("crate.jpg", ))

camera = Camera(
    width=400, height=400,
    pos=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),   # looking along +Z
    d=300
)
cube = Cube()
color = [np.array((0,1,0)), np.array((0,1,0)), np.array((0,0,1)), np.array((0,0,1)), np.array((1,1,0)), np.array((1,1,0)), np.array((0,1,1)), np.array((0,1,1)), np.array((1,0,0)), np.array((1,0,0)), np.array((1,0,1)), np.array((1,0,1))]
cube.set_colors(color)
t = time()
red_crate_shader = Unlit_Texture(crate_atlas, [1, 0, 0])
green_crate_shader = Unlit_Texture(crate_atlas, [0, 1, 0])
blue_crate_shader = Unlit_Texture(crate_atlas, [0, 0, 1])
crate_shader = Unlit_Texture(crate_atlas)

objects = [
    #Instance(cube, Transform(Point3D(0, 0, 3), Point3D(2, 2, 2), Point3D(0, 28, 13))),
    #Instance(cube, Transform(Point3D(0, 0, 5), Point3D(4, 4, 4), Point3D(14, 180, 0))),
    Instance(cube, Transform(Point3D(0, 0, 2), Point3D(1, 1, 1), Point3D(45, 45, 90)), crate_shader),
    Instance(cube, Transform(Point3D(0, 0, 2), Point3D(0.5, 0.5, 1), Point3D(0, 25, 15)), crate_shader),
    Instance(cube, Transform(Point3D(1, 2, 5), Point3D(2, 1, 1), Point3D(0, 0, 45)), crate_shader),
    Instance(cube, Transform(Point3D(3, -2, 5), Point3D(1, 4, 1), Point3D(15, 0, 0)), crate_shader),
    Instance(cube, Transform(Point3D(-10, -8, 20), Point3D(10, 10, 10), Point3D(-30, 90, 15)), crate_shader)
]

post_process = [PPFog(500, [0, 0, 0])]

viewport = Viewport(objects, camera, perspective_projection)
pyplot.imsave(f"img/image_wireframe.png", render_wireframe(viewport, 500, 500))
r, d_b = render(viewport, 400, 400, 500, post_process)
uv = np.zeros((len(d_b), len(d_b[0]), 3))
for y in range(len(d_b[0])):
    for x in range(len(d_b)):
        uv[x, y, :] = [d_b[x, y, 4], d_b[x, y, 5], 0]
pyplot.imsave(f"img/image.png", r)
pyplot.imsave(f"img/normal.png", d_b[:, :, :3])
pyplot.imsave(f"img/depth.png", d_b[:, :, 3])
pyplot.imsave(f"img/uv.png", uv)
print(f"Done — saved image.png in {time()-t} seconds")
