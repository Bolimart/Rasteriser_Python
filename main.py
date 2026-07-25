from matplotlib import pyplot
from renderer import *
from time import time
from lights import *

#——————————[ ASSETS ]————————————————————————————————————————————————————————————————————————————————————————————————————

grid_atlas = Atlas(pyplot.imread("grid.png"))
crate_atlas = Atlas(pyplot.imread("crate.png"))
tourniquet = load_obj('models/utah_teapot_reso2.obj', smooth_shading=True)
#pilier = load_obj('models/tourniquet_pilier.obj', smooth_shading=True)

#——————————[ SCENE SETUP ]———————————————————————————————————————————————————————————————————————————————————————————————

camera = Camera(
    width  = 400,
    height = 400,
    pos    = (0.0, 0.0, 0.0),
    normal = (0.0, 0.0, 1.0),  # looking along +Z
    d      = 300,
    view_dist= 500,
    gamma = 1.5
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

lights = [
    PointLight(np.array([5, 5, 0])),
    PointLight(np.array([-5, 0, 0]), np.array([0.05, 0.01, 0.03]), np.array([0.5, 0.1, 0.3]), np.array([0.5, 0.1, 0.3]), intensity=0.4),
]

# Materials
grid_shader   = LitTexture(grid_atlas, lights,True, np.array([0.05, 0.05, 0.05]), np.array([0.7, 0.7, 0.7]), np.array([1, 1, 1]), 50, 1)
crate_shader  = LitTexture(crate_atlas, lights, aliasing=True)

# Scene objects
objects = [
    make_instance(tourniquet, Transform(Point3D(0, -2, 5.5), Point3D(1, 1, 1), Point3D(-100, -12, 0)), grid_shader),
]

post_process = [
    PPFog(1200, [0, 0, 0]),  # black fog, fully opaque at distance 800
]

#——————————[ RENDER ]————————————————————————————————————————————————————————————————————————————————————————————————————

viewport = Viewport(objects, camera, perspective_projection)

t = time()

print("Rendering wireframe:")
pyplot.imsave("img/image_wireframe.png", render_wireframe(viewport, 4000, 4000))
print(f"Done — saved wireframe.png in {time() - t:.3f} seconds")

print("Rendering image:")
r, d_b = render(viewport, 400, 400)

# Build a UV visualisation image from G-buffer channels 4 and 5
uv = np.zeros((len(d_b), len(d_b[0]), 3))
for y in range(len(d_b[0])):
    for x in range(len(d_b)):
        uv[x, y, :] = [d_b[x, y, 4], d_b[x, y, 5], 0]

pyplot.imsave("img/image.png",  r)
pyplot.imsave("img/normal.png", d_b[:, :, :3])
pyplot.imsave("img/depth.png",  d_b[:, :, 3])
pyplot.imsave("img/uv.png",     np.clip(uv, 0, 1))

print(f"Done — saved image.png in {time() - t:.3f} seconds")