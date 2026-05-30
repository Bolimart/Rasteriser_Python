from matplotlib import pyplot
from renderer import *
from time import time

#——————————[ ASSETS ]————————————————————————————————————————————————————————————————————————————————————————————————————

crate_atlas = Atlas(pyplot.imread("crate.png"))

#——————————[ SCENE SETUP ]———————————————————————————————————————————————————————————————————————————————————————————————

camera = Camera(
    width  = 400,
    height = 400,
    pos    = (0.0, 0.0, 0.0),
    normal = (0.0, 0.0, 1.0),  # looking along +Z
    d      = 300,
)

# Shared geometry — all instances reference the same Cube model
cube = Cube()
cube.set_colours([
    np.array((0, 1, 0)), np.array((0, 1, 0)),  # front
    np.array((0, 0, 1)), np.array((0, 0, 1)),  # back
    np.array((1, 1, 0)), np.array((1, 1, 0)),  # left
    np.array((0, 1, 1)), np.array((0, 1, 1)),  # right
    np.array((1, 0, 0)), np.array((1, 0, 0)),  # bottom
    np.array((1, 0, 1)), np.array((1, 0, 1)),  # top
])

# Materials
red_crate_shader   = UnlitTexture(crate_atlas, [1, 0, 0])
green_crate_shader = UnlitTexture(crate_atlas, [0, 1, 0])
blue_crate_shader  = UnlitTexture(crate_atlas, [0, 0, 1])
crate_shader       = UnlitTexture(crate_atlas)

# Scene objects
objects = [
    # Instance(cube, Transform(Point3D(0, 0, 3), Point3D(2, 2, 2), Point3D(0,  28,  13))),
    # Instance(cube, Transform(Point3D(0, 0, 5), Point3D(4, 4, 4), Point3D(14, 180,  0))),
    Instance(cube, Transform(Point3D( 0,  0,  2), Point3D(1,   1,   1), Point3D( 45, 45,  90)), crate_shader),
    Instance(cube, Transform(Point3D( 0,  0,  2), Point3D(0.5, 0.5, 1), Point3D(  0, 25,  15)), blue_crate_shader),
    Instance(cube, Transform(Point3D( 1,  2,  5), Point3D(2,   1,   1), Point3D(  0,  0,  45)), red_crate_shader),
    Instance(cube, Transform(Point3D( 3, -2,  5), Point3D(1,   4,   1), Point3D( 15,  0,   0)), blue_crate_shader),
    Instance(cube, Transform(Point3D(-10,-8, 20), Point3D(10, 10,  10), Point3D(-30, 90,  15)), crate_shader),
]

post_process = [
    PPFog(800, [0, 0, 0]),  # black fog, fully opaque at distance 800
]

#——————————[ RENDER ]————————————————————————————————————————————————————————————————————————————————————————————————————

viewport = Viewport(objects, camera, perspective_projection)

t = time()

pyplot.imsave("img/image_wireframe.png", render_wireframe(viewport, 400, 400))

r, d_b = render(viewport, 400, 400, 500, post_process)

# Build a UV visualisation image from G-buffer channels 4 and 5
uv = np.zeros((len(d_b), len(d_b[0]), 3))
for y in range(len(d_b[0])):
    for x in range(len(d_b)):
        uv[x, y, :] = [d_b[x, y, 4], d_b[x, y, 5], 0]

pyplot.imsave("img/image.png",  r)
pyplot.imsave("img/normal.png", d_b[:, :, :3])
pyplot.imsave("img/depth.png",  d_b[:, :, 3])
pyplot.imsave("img/uv.png",     uv)

print(f"Done — saved image.png in {time() - t:.3f} seconds")