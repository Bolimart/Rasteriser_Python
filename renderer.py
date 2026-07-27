from math import floor, ceil
from multiprocessing import Pool, get_context
import numpy as np
from time import time
from objects import *
from viewport import *

#——————————[ UTILITIES ]————————————————————————————————————————————————————————————————————————————————————————————————

def swap(x0: "any", x1: "any") -> tuple:
    """
    Swap two values.

    :param x0: First value
    :param x1: Second value
    :return:   (x1, x0)
    """
    return x1, x0


def lerp(a: float, b: float, t: float) -> float:
    """
    Linearly interpolate between a and b by factor t.

    :param a: Start value (returned when t=0)
    :param b: End value (returned when t=1)
    :param t: Interpolation factor in [0, 1]
    :return:  a + t * (b - a)
    """
    return a + t * (b - a)


def interp(i0: float, d0: float, i1: float, d1: float) -> np.array:
    """
    Interpolate integer values of d along the integer range [i0, i1].

    Used for rasterising X coordinates along triangle edges — values are rounded
    to integers. Do not use for depth or UV interpolation (use interp_float instead).

    :param i0: Start index (scanline or column)
    :param d0: Value at i0
    :param i1: End index
    :param d1: Value at i1
    :return:   List of rounded integer values, one per step from i0 to i1 inclusive
    """
    i0, i1 = int(i0), int(i1)
    if i0 == i1:
        return np.array([d0])
    return np.round(np.linspace(d0, d1, i1 - i0 + 1)).astype(int)


#——————————[ RASTERISATION ]—————————————————————————————————————————————————————————————————————————————————————————————


def draw_triangle(triangle: Triangle, data_buffer: np.ndarray, view_dist: float, mat_id: int) -> None:

    width, height, _ = data_buffer.shape

    x0, y0, z0 = triangle.P0
    x1, y1, z1 = triangle.P1
    x2, y2, z2 = triangle.P2

    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    if area == 0:
        return

    u0, v0 = triangle.P0_uv
    u1, v1 = triangle.P1_uv
    u2, v2 = triangle.P2_uv
    u0, v0 = u0 * z0, v0 * z0
    u1, v1 = u1 * z1, v1 * z1
    u2, v2 = u2 * z2, v2 * z2,
    n0 = triangle.P0_n
    n1 = triangle.P1_n
    n2 = triangle.P2_n

    smooth = True
    if (n0[0] == n1[0] and n0[0] == n2[0]) and (n0[1] == n1[1] and n0[1] == n2[1]) and (
            n0[2] == n1[2] and n0[2] == n2[2]):
        smooth = False

    # Define the bounding box, clamped to the canva size
    x_max = np.clip(ceil(max(x0, x1, x2)), 0, width-1)
    x_min = np.clip(floor(min(x0, x1, x2)), 0, width-1)
    y_max = np.clip(ceil(max(y0, y1, y2)), 0, height-1)
    y_min = np.clip(floor(min(y0, y1, y2)), 0, height-1)

    # Create a matrix of all the pixels in the bounding box
    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    px, py = np.meshgrid(xs, ys)

    e0 = (x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)
    e1 = (x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)
    e2 = (x0 - px) * (y1 - py) - (x1 - px) * (y0 - py)

    # Check if each pixel is inside the triangle -> boolean mask of the pixels inside
    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))

    # Barycentric weights
    w0 = e0 / area
    w1 = e1 / area
    w2 = 1.0 - w0 - w1

    inv_z = w0 * z0 + w1 * z1 + w2 * z2 # depth mask

    region = data_buffer[y_min:y_max + 1, x_min:x_max + 1] # The region of the DB inside the bounding box
    write = inside & (inv_z > region[: , :, 3]) # The pixels that are inside the triangle and above the last depth value

    u = (w0 * u0 + w1 * u1 + w2 * u2) / inv_z
    v = (w0 * v0 + w1 * v1 + w2 * v2) / inv_z

    if smooth: # Smooth normals
        # each component is a 2D array; stack into (H_box, W_box, 3)
        n = np.stack([
            w0 * n0[0] + w1 * n1[0] + w2 * n2[0],
            w0 * n0[1] + w1 * n1[1] + w2 * n2[1],
            w0 * n0[2] + w1 * n1[2] + w2 * n2[2],
        ], axis=-1) # Stack the normal vectors along the third axis, axis=-1 means stack along the last axis
        length = np.linalg.norm(n, axis=-1, keepdims=True) # Get the length of each normal vector, keepdims to keep the same shape as n
        length[length == 0] = 1.0 # Avoid division by zero
        normal_encoded = (n / length + 1) * 0.5

    # Write the triangle's data into the G-buffer'
    region[write, 3] = inv_z[write] # Write only where write is True
    region[write, 4] = u[write]
    region[write, 5] = v[write]
    region[write, 6] = mat_id
    if smooth:
        region[write, 0:3] = normal_encoded[write]
    else:
        region[write, 0:3] = (np.asarray(triangle.P0_n) + 1) * 0.5


def draw_wireframe_triangle(triangle: Triangle, image: np.ndarray) -> None:
    """
    Draw the three edges of a triangle onto the image using the triangle's colour.

    :param triangle: Triangle to draw
    :param image:    Image array of shape (height, width, 3)
    """
    l = triangle.lines()
    draw_line(l[0], triangle.colour, image)
    draw_line(l[1], triangle.colour, image)
    draw_line(l[2], triangle.colour, image)


def draw_line(line: 'Line', colour: list, image: np.ndarray) -> None:
    """
    Draw a line segment onto the image using Bresenham-style integer interpolation.

    The line is rasterised along its dominant axis (X for shallow lines, Y for steep),
    avoiding gaps that would appear if stepping along the shorter axis.

    :param line:  Line object with P0 and P1 endpoints
    :param colour: RGB colour in [0, 1]
    :param image: Image array of shape (height, width, 3)
    """
    h, w = image.shape[:2]
    x0, y0 = int(line.P0[0]), int(line.P0[1])
    x1, y1 = int(line.P1[0]), int(line.P1[1])

    # Shallow line — step along X, interpolate Y
    if abs(x1 - x0) > abs(y1 - y0):
        if x0 > x1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        ys = interp(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            if 0 <= x < w and 0 <= ys[x - x0] < h:
                image[ys[x - x0], x] = colour

    # Steep line — step along Y, interpolate X
    else:
        if y0 > y1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        xs = interp(y0, x0, y1, x1)
        for y in range(y0, y1 + 1):
            if 0 <= y < h and 0 <= xs[y - y0] < w:
                image[y, xs[y - y0]] = colour

#——————————[ SHADING ]————————————————————————————————————————————————————————————————————————————————————————————————————

def draw_pixel(material: 'Material', image: np.ndarray, data_buffer: np.ndarray, x: int, y: int, camera: 'Camera') -> None:
    """
    Shade a single pixel by calling the material's get_pixel() and compositing the
    result onto the image according to the material's blend mode.

    Blend modes:
      BLEND_OPAQUE   — overwrite the pixel with the material colour
      BLEND_ALPHA    — composite using alpha: dst = src*a + dst*(1-a)
      BLEND_ADDITIVE — add the material colour to the pixel, clamped to [0, 1]
      BLEND_MULTIPLY — multiply the pixel by the material colour, clamped to [0, 1]

    :param material:    Material whose get_pixel() provides the shaded colour
    :param image:       Image array of shape (height, width, 3) — modified in place
    :param data_buffer: G-buffer array of shape (height, width, 7) — read only
    :param x:           Pixel column
    :param y:           Pixel row
    :param camera:      Camera passed through to the material's get_pixel()
    """
    blend = (material.id >> 0) & 0x3

    if blend == BLEND_OPAQUE:
        image[x, y] = material.get_pixel(image, data_buffer, x, y, camera)

    elif blend == BLEND_ALPHA:
        c = np.array(material.get_pixel(image, data_buffer, x, y, camera), dtype=float)
        image[x, y] = image[x, y] * (1 - c[3]) + c[:3] * c[3]

    elif blend == BLEND_ADDITIVE:
        c = np.array(material.get_pixel(image, data_buffer, x, y, camera), dtype=float)
        image[x, y] = np.clip(image[x, y] + c[:3], 0, 1)

    elif blend == BLEND_MULTIPLY:
        c = np.array(material.get_pixel(image, data_buffer, x, y, camera), dtype=float)
        image[x, y] = np.clip(image[x, y] * c[:3], 0, 1)

#——————————[ PROJECTION ]————————————————————————————————————————————————————————————————————————————————————————————————

def perspective_projection(triangle: Triangle, camera: 'Camera', width: int, height: int) -> Triangle:
    """
    Project a world-space triangle onto the screen using perspective projection.

    Each vertex is projected from 3D view space to 2D screen space using the
    pinhole camera model: x_screen = x * d/z, y_screen = y * d/z.
    The original z value is stored as 1/z in the projected vertex's third component
    for use in perspective-correct depth testing and UV interpolation.

    Degenerate triangles (where all three projected vertices are within 10e-3 of each
    other) are discarded and None is returned.

    :param triangle: Triangle in world space
    :param camera:   Camera defining d, width, height, and position
    :param width:    Output image width in pixels
    :param height:   Output image height in pixels
    :return:         Projected Triangle in screen space, or None if degenerate
    """
    def viewport_to_canvas(x: float, y: float, z: float) -> np.ndarray:
        """Map from viewport units to pixel coordinates, storing 1/z as depth."""
        cx = width  / 2 + x * (width  / camera.width)
        cy = height / 2 - y * (height / camera.height)  # flip Y so +Y is up in world space
        return Point3D(cx, cy, 1.0 / z)

    def project_vertex(v: np.ndarray) -> np.ndarray:
        """Apply perspective divide and map to canvas coordinates."""
        return viewport_to_canvas(v[0] * (camera.d / v[2]),
                                  v[1] * (camera.d / v[2]),
                                  v[2])

    P0 = project_vertex(triangle.P0)
    P1 = project_vertex(triangle.P1)
    P2 = project_vertex(triangle.P2)

    return Triangle(P0, P1, P2, triangle.colour, triangle.P0_uv, triangle.P1_uv, triangle.P2_uv, triangle.P0_n, triangle.P1_n, triangle.P2_n)

#——————————[ RENDER ]————————————————————————————————————————————————————————————————————————————————————————————————————

def render_wireframe_instance(instance: Instance, viewport: 'Viewport', image: np.ndarray, width: int, height: int, DEBUG=False) -> None:
    """
    Render a single instance as a wireframe by drawing the edges of each visible triangle.

    :param instance: Instance to render
    :param viewport: Viewport containing the camera and projection function
    :param image:    Image array of shape (height, width, 3) — modified in place
    :param width:    Output image width in pixels
    :param height:   Output image height in pixels
    """
    if DEBUG:
        t_len = len(instance.model.triangles)
        i = 1
    for t in instance.apply_transform():
        projected = viewport.project_func(t, viewport.camera, width, height)
        draw_wireframe_triangle(projected, image)
        if DEBUG:
            print(f"{i/t_len*100}")
            i += 1


def render_instance(instance: Instance, viewport: 'Viewport', data_buffer: np.ndarray, width: int, height: int, DEBUG=False) -> None:
    """
    Rasterise a single instance into the G-buffer.

    Applies the instance transform, culls back-facing and out-of-frustum triangles,
    projects the survivors to screen space, and writes them into the G-buffer via
    draw_triangle().

    The double-sided flag is read from the material ID to decide whether to skip
    back-face culling for this instance.

    :param instance:   Instance to render
    :param viewport:   Viewport containing the camera and projection function
    :param data_buffer: G-buffer array of shape (height, width, 7) — modified in place
    :param width:      Output image width in pixels
    :param height:     Output image height in pixels
    """
    draw_back_faces = (instance.mat.id >> 3) & 0x1  # read double-sided flag from material ID
    if DEBUG:
        t_len = len(instance.model.triangles)
        i = 1
    for t in instance.apply_transform():
        if viewport.camera.in_view(t, draw_back_faces):
            projected = viewport.project_func(t, viewport.camera, width, height)
            draw_triangle(projected, data_buffer, viewport.camera.view_dist, instance.mat.id)
        if DEBUG:
            print(f"{i/t_len*100}")
            i += 1


def render(viewport, width, height, post_process=[], benchmark_data=None, DEBUG=False):
    t_total = time()

    materials   = MaterialRegistry()
    image       = np.zeros((height, width, 3))
    data_buffer = np.zeros((height, width, 7))
    i = 0

    # ── Pass 1 — Geometry ──────────────────────────────────────────────────
    t = time()
    for instance in viewport.objects:
        if type(instance) is ComplexInstance:
            for inst_ in instance.instances:
                materials.register(inst_.mat)
                render_instance(inst_, viewport, data_buffer, width, height, DEBUG)
                i += 1
        else:
            materials.register(instance.mat)
            render_instance(instance, viewport, data_buffer, width, height, DEBUG)
            i += 1
    if benchmark_data is not None:
        benchmark_data["time_pass1"] = round(time() - t, 4)

    # ── Pass 2 — Shading ───────────────────────────────────────────────────
    t = time()
    pixels_shaded = 0
    for x in range(len(data_buffer)):
        for y in range(len(data_buffer[0])):
            material_data = int(data_buffer[x, y, 6])
            if material_data != 0:
                pixels_shaded += 1
                for k in range(material_data.bit_length() // 16 + 1):
                    material_id = material_data >> 16 * k
                    material    = materials.get(material_id)
                    draw_pixel(material, image, data_buffer, x, y, viewport.camera)
    if benchmark_data is not None:
        benchmark_data["time_pass2"]   = round(time() - t, 4)
        benchmark_data["pixels_shaded"] = pixels_shaded

    # ── Pass 3 — Post-process + gamma ──────────────────────────────────────
    t = time()
    if post_process:
        for x in range(len(data_buffer)):
            for y in range(len(data_buffer[0])):
                for pp in post_process:
                    draw_pixel(pp, image, data_buffer, x, y, viewport.camera)
    image = image ** (1 / viewport.camera.gamma)
    if benchmark_data is not None:
        benchmark_data["time_pass3"] = round(time() - t, 4)

    if benchmark_data is not None:
        benchmark_data["time_total"] = round(time() - t_total, 4)

    return image, data_buffer

# TODO: Camera rotation


def render_wireframe(viewport: 'Viewport', width: int, height: int, DEBUG:bool=False) -> np.ndarray:
    """
    Render the full scene as a wireframe.

    :param viewport: Viewport containing the scene objects, camera, and projection function
    :param width:    Output image width in pixels
    :param height:   Output image height in pixels
    :return:         Image array of shape (height, width, 3)
    """
    image = np.zeros((height, width, 3))
    i = 0
    for instance in viewport.objects:
        if type(instance) is ComplexInstance:
            for inst in instance.instances:
                print(f"Rendreing instance {i} with {len(inst.model.triangles)} triangles")
                render_wireframe_instance(inst, viewport, image, width, height)
                i += 1
        else:
            print(f"Rendreing instance {i} with {len(instance.model.triangles)} triangles")
            render_wireframe_instance(instance, viewport, image, width, height)
            i += 1

    return image