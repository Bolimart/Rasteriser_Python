from math import floor

import numpy as np
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


def interp(i0: float, d0: float, i1: float, d1: float) -> list:
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
    i0 = np.int64(i0)
    i1 = np.int64(i1)
    if i0 == i1:
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i0, i1 + 1):
        values.append(round(d))
        d = d + a
    return values


def interp_float(i0: float, d0: float, i1: float, d1: float) -> list:
    """
    Interpolate float values of d along the integer range [i0, i1].

    Used for depth and UV interpolation along triangle edges, where rounding
    would destroy the precision needed for correct depth testing and texture mapping.

    :param i0: Start index (scanline or column)
    :param d0: Float value at i0
    :param i1: End index
    :param d1: Float value at i1
    :return:   List of float values, one per step from i0 to i1 inclusive
    """
    i0, i1 = np.int64(i0), np.int64(i1)
    if i0 == i1:
        return [float(d0)]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i0, i1 + 1):
        values.append(float(d))
        d += a
    return values

#——————————[ RASTERISATION ]—————————————————————————————————————————————————————————————————————————————————————————————

def draw_triangle(triangle: Triangle, data_buffer: np.ndarray, view_dist: float, mat_id: int) -> None:
    """
    Rasterise a projected triangle into the G-buffer using a scanline algorithm.

    Writes depth, normal, UV coordinates, and material ID per pixel.
    Uses a depth test to keep only the closest surface at each pixel.

    UV coordinates are interpolated perspective-correctly by interpolating u/z and v/z
    along edges (which are linear in screen space), then dividing by 1/z at each pixel
    to recover the true u and v values.

    G-buffer channels written:
        [0:3] — world-space normal, encoded as (n + 1) * 0.5
        [3]   — depth as 1/z, normalised by view_dist to [0, 1]
        [4:6] — perspective-correct UV coordinates (u, v)
        [6]   — material ID

    :param triangle:   Projected Triangle with screen-space P0/P1/P2 and 1/z depth
    :param data_buffer: G-buffer array of shape (height, width, 7)
    :param view_dist:  Maximum view distance used to normalise depth to [0, 1]
    :param mat_id:     Material ID to write into the G-buffer for this triangle
    """
    x0, y0, depth0 = triangle.P0
    x1, y1, depth1 = triangle.P1
    x2, y2, depth2 = triangle.P2
    u0, v0 = triangle.P0_uv
    u1, v1 = triangle.P1_uv
    u2, v2 = triangle.P2_uv

    # Premultiply UVs by depth (1/z) so they interpolate linearly in screen space.
    # Dividing by the interpolated 1/z at each pixel recovers the perspective-correct value.
    u0, v0 = u0 * depth0, v0 * depth0
    u1, v1 = u1 * depth1, v1 * depth1
    u2, v2 = u2 * depth2, v2 * depth2

    # Step 1 — Sort vertices so that y0 <= y1 <= y2 (top to bottom).
    # All associated per-vertex data (depth, u, v) is swapped alongside x and y.
    if y1 < y0: x0,x1=swap(x0,x1); y0,y1=swap(y0,y1); depth0,depth1=swap(depth0,depth1); u0,u1=swap(u0,u1); v0,v1=swap(v0,v1)
    if y2 < y0: x0,x2=swap(x0,x2); y0,y2=swap(y0,y2); depth0,depth2=swap(depth0,depth2); u0,u2=swap(u0,u2); v0,v2=swap(v0,v2)
    if y2 < y1: x1,x2=swap(x1,x2); y1,y2=swap(y1,y2); depth1,depth2=swap(depth1,depth2); u1,u2=swap(u1,u2); v1,v2=swap(v1,v2)

    # Step 2 — Interpolate X, depth, u, and v along all three edges.
    # x01/x12/x02 are integer X values per scanline; depth/u/v are floats.
    # The long edge P0→P2 is always computed. The two short edges P0→P1 and P1→P2
    # are computed separately and concatenated to cover the same scanline range.
    x01 = interp(y0, x0, y1, x1);       x12 = interp(y1, x1, y2, x2);       x02 = interp(y0, x0, y2, x2)
    u01 = interp_float(y0, u0, y1, u1); u12 = interp_float(y1, u1, y2, u2); u02 = interp_float(y0, u0, y2, u2)
    v01 = interp_float(y0, v0, y1, v1); v12 = interp_float(y1, v1, y2, v2); v02 = interp_float(y0, v0, y2, v2)
    d01 = interp_float(y0, depth0, y1, depth1); d12 = interp_float(y1, depth1, y2, depth2); d02 = interp_float(y0, depth0, y2, depth2)

    # Step 3 — Concatenate the two short-edge lists into one list spanning P0→P1→P2.
    # The shared middle vertex (P1) appears at the end of x01 and the start of x12,
    # so pop the duplicate before concatenating.
    x01.pop(); x012 = x01 + x12; del x01, x12
    d01.pop(); d012 = d01 + d12; del d01, d12
    u01.pop(); u012 = u01 + u12; del u01, u12
    v01.pop(); v012 = v01 + v12; del v01, v12

    # Step 4 — Determine which edge list is the left boundary and which is the right.
    # Sample the middle scanline of both lists: whichever has the smaller X is on the left.
    m = floor(len(x012) / 2)
    if x02[m] < x012[m]:
        x_left, x_right = x02, x012
        d_left, d_right = d02, d012
        u_left, u_right = u02, u012
        v_left, v_right = v02, v012
    else:
        x_left, x_right = x012, x02
        d_left, d_right = d012, d02
        u_left, u_right = u012, u02
        v_left, v_right = v012, v02
    del x02, x012, d02, d012, u02, u012, v02, v012

    # Step 5 — Fill horizontal scanlines between the left and right edges.
    height, width = data_buffer.shape[:2]
    y0, y2 = int(y0), int(y2)

    # Encode the normal once per triangle — the same value is written for every pixel
    normal_encoded = [
        (triangle.P2_n[0] + 1) * 0.5,
        (triangle.P2_n[1] + 1) * 0.5,
        (triangle.P2_n[2] + 1) * 0.5,
    ]

    for y in range(y0, y2):
        i = y - y0
        xl, xr = int(x_left[i]), int(x_right[i])
        dl, dr = d_left[i], d_right[i]
        ul, ur = u_left[i], u_right[i]
        vl, vr = v_left[i], v_right[i]

        for x in range(xl, xr):

            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # Interpolate 1/z and premultiplied u/z, v/z across the scanline.
            t = (x - xl) / (xr - xl) if xr != xl else 0
            d         = float(lerp(dl, dr, t))
            inv_depth = min(max(d / view_dist, 0), 1) if d != 0 else 0

            # Depth test — skip pixel if something closer is already written
            if data_buffer[y, x, 3] >= inv_depth:
                continue

            # Recover perspective-correct u and v by dividing out the premultiplied 1/z
            u = lerp(ul, ur, t) / d if d != 0 else 0
            v = lerp(vl, vr, t) / d if d != 0 else 0

            data_buffer[y, x, 0:3] = normal_encoded  # encoded world-space normal
            data_buffer[y, x, 3]   = inv_depth        # normalised depth (1/z)
            data_buffer[y, x, 4:6] = [u, v]           # perspective-correct UV
            data_buffer[y, x, 6]   = mat_id           # material ID

        # TODO : Add smooth normal interpolation


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

def render_wireframe_instance(instance: Instance, viewport: 'Viewport', image: np.ndarray, width: int, height: int) -> None:
    """
    Render a single instance as a wireframe by drawing the edges of each visible triangle.

    :param instance: Instance to render
    :param viewport: Viewport containing the camera and projection function
    :param image:    Image array of shape (height, width, 3) — modified in place
    :param width:    Output image width in pixels
    :param height:   Output image height in pixels
    """
    for t in instance.apply_transform():
        projected = viewport.project_func(t, viewport.camera, width, height)
        if projected is not None:
            draw_wireframe_triangle(projected, image)


def render_instance(instance: Instance, viewport: 'Viewport', data_buffer: np.ndarray, width: int, height: int, view_dist: float) -> None:
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
    :param view_dist:  Maximum view distance used to normalise depth values
    """
    draw_back_faces = (instance.mat.id >> 3) & 0x1  # read double-sided flag from material ID
    for t in instance.apply_transform():
        if viewport.camera.in_view(t, draw_back_faces):
            T = viewport.project_func(t, viewport.camera, width, height)
            if T is not None:
                draw_triangle(T, data_buffer, view_dist, instance.mat.id)


def render(viewport: 'Viewport', width: int, height: int, view_dist: float, post_process: list = []) -> tuple:
    """
    Render the full scene to an RGB image.

    The render pipeline runs in three passes:

      1. Geometry pass — rasterise all instances into the G-buffer (depth, normals, UVs,
         material IDs). No shading is done here.

      2. Shading pass — iterate over every pixel, read the material ID from the G-buffer,
         and call the corresponding material's get_pixel() to produce the final colour.
         Materials are processed in the order they were registered.

      3. Post-process pass — apply each post-process effect in order. Effects read from
         the G-buffer and composite onto the already-shaded image.

    :param viewport:     Viewport containing the scene objects, camera, and projection function
    :param width:        Output image width in pixels
    :param height:       Output image height in pixels
    :param view_dist:    Maximum view distance for depth normalisation
    :param post_process: Ordered list of PostProcess materials to apply after shading
    :return:             Tuple of (image, data_buffer) — both np.ndarray
    """
    materials    = MaterialRegistry()
    image        = np.zeros((height, width, 3))
    data_buffer  = np.zeros((height, width, 7))

    # Pass 1 — Geometry: rasterise all instances into the G-buffer
    for instance in viewport.objects:
        if type(instance) is ComplexInstance:
            for inst in instance.instances:
                materials.register(inst.mat)
                render_instance(inst, viewport, data_buffer, width, height, view_dist)
        else:
            materials.register(instance.mat)
            render_instance(instance, viewport, data_buffer, width, height, view_dist)

    # Pass 2 — Shading: resolve each pixel's material and shade it
    for x in range(len(data_buffer)):
        for y in range(len(data_buffer[0])):
            material_data = int(data_buffer[x, y, 6])
            if material_data != 0:
                for i in range(material_data.bit_length() // 16 + 1):
                    material_id = material_data >> 16 * i
                    material    = materials.get(material_id)
                    draw_pixel(material, image, data_buffer, x, y, viewport.camera)

    # Pass 3 — Post-process: apply screen-space effects in order
    for x in range(len(data_buffer)):
        for y in range(len(data_buffer[0])):
            for pp in post_process:
                draw_pixel(pp, image, data_buffer, x, y, viewport.camera)

    return image, data_buffer

# TODO: Camera rotation


def render_wireframe(viewport: 'Viewport', width: int, height: int) -> np.ndarray:
    """
    Render the full scene as a wireframe.

    :param viewport: Viewport containing the scene objects, camera, and projection function
    :param width:    Output image width in pixels
    :param height:   Output image height in pixels
    :return:         Image array of shape (height, width, 3)
    """
    image = np.zeros((height, width, 3))
    for instance in viewport.objects:
        render_wireframe_instance(instance, viewport, image, width, height)
    return image