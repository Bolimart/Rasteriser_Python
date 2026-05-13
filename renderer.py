from math import floor

import numpy
import numpy as np
from objects import *
from viewport import *


# =====[ Utilities ]=====

def swap (x0, x1):
    return x1, x0

def lerp(i0, d0, i1, d1):
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

# =====[ Draw Functions ]=====

def draw_triangle_dot(triangle, image):
    # This function go through the entire image, which isn't the best
    # Also, some border pixels are not drawn.
    lines = triangle.lines()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            p = Point2D(x, y)
            if triangle.is_point_inside(p):
                image[y, x] = triangle.color

    for line in lines:
        draw_line(line, [1, 1, 1], image)


def draw_triangle(triangle, image, data_buffer):
    
    # This one is way more optimized, but less straight forward
    x0, y0, depth0 = triangle.P0
    x1, y1, depth1 = triangle.P1
    x2, y2, depth2 = triangle.P2

    # 1 - Sort the points so that y0 <= y1 <= y2
    if y1 < y0:
        x0, x1 = swap(x0, x1)
        y0, y1 = swap(y0, y1)
    if y2 < y0:
        x0, x2 = swap(x0, x2)
        y0, y2 = swap(y0, y2)
    if y2 < y1:
        x1, x2 = swap(x1, x2)
        y1, y2 = swap(y1, y2)
    # The "tall" side (the side with the > Y0 - Y1 is always P0 to P2

    # 2 - Compute the x coordinates of the triangle edges
    x01 = lerp(y0, x0, y1, x1)
    x12 = lerp(y1, x1, y2, x2)
    x02 = lerp(y0, x0, y2, x2)

    # 3 - Concatenate the short sides
    x01.pop() # Remove the last coordinate of x01 as it is the same as the first of x12
    x012 = x01 + x12
    del(x01, x12)

    # 4 - Determine which is left and which is right
    m = floor(len(x012) / 2)
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
    else:
        x_left = x012
        x_right = x02
    del(x02, x012)

    # 5 - Draw the horizontal segments
    y0 = np.int64(y0)
    y2 = np.int64(y2)
    for y in range(y0, y2):
        for x in range(x_left[y - y0], x_right[y - y0]):
            if x < 0 or x >= len(image) or y < 0 or y >= len(image[0]):
                continue
            image[y, x] = triangle.color


def draw_wireframe_triangle(triangle: Triangle, image):
    l = triangle.lines()
    draw_line(l[0], triangle.color, image)
    draw_line(l[1], triangle.color, image)
    draw_line(l[2], triangle.color, image)


def draw_line(line, color, image):
    h, w = image.shape[:2]
    x0, y0 = int(line.P0[0]), int(line.P0[1])
    x1, y1 = int(line.P1[0]), int(line.P1[1])

    # The line is horizontal-ish
    if abs(x1 - x0) > abs(y1 - y0):
        if x0 > x1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        ys = lerp(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            if 0 <= x < w and 0 <= ys[x - x0] < h:
                image[ys[x - x0], x] = color

    # The line is vertical-ish
    else:
        if y0 > y1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        xs = lerp(y0, x0, y1, x1)
        for y in range(y0, y1 + 1):
            if 0 <= y < h and 0 <= xs[y - y0] < w:
                image[y, xs[y - y0]] = color

# =====[ Projections ]=====

def perspective_projection(triangle: Triangle, camera: Camera, width, height):
    def viewport_to_canvas(x, y, z):
        cx = width / 2 + x * (width / camera.width)
        cy = height / 2 - y * (height / camera.height)  # flip Y so +Y is up
        depth = np.linalg.norm(Point3D(x, y, z) - camera.pos)
        return Point3D(cx, cy, depth)
    
    def project_vertex(v):
        return viewport_to_canvas(v[0] * (camera.d / v[2]), v[1] * (camera.d / v[2]), v[2])

    P0 = project_vertex(triangle.P0)
    P1 = project_vertex(triangle.P1)
    P2 = project_vertex(triangle.P2)
    
    return Triangle(P0, P1, P2, triangle.color)

# =====[ Render ]=====

def render_wireframe_instance(instance:Instance, viewport: Viewport, image, width, height):
    projected = []
    for t in instance.apply_transform():
        projected.append(viewport.project_func(t, viewport.camera, width, height))
    for T in projected:
        draw_wireframe_triangle(T, image)
    

def render_instance(instance:Instance, viewport: Viewport, image, data_buffer, width, height):
    projected = []
    for t in instance.apply_transform():
        if viewport.camera.in_view(t):
            projected.append(viewport.project_func(t, viewport.camera, width, height))
    for t in instance.apply_transform()[8:10]:
        normal = t.get_normal()
        to_camera = viewport.camera.pos - t.P0
        dot = np.dot(normal, to_camera)
        viewport.camera.in_view(t, True)
    for T in projected:
        to_camera = viewport.camera.pos - T.P0
        print(T.P0, T.P1, T.P2, T.normal)
        draw_triangle(T, image, data_buffer)


def render(viewport:Viewport, width, height):

    image = numpy.zeros((height, width, 3))
    data_buffer = numpy.zeros((height, width, 4))

    for instance in viewport.objects:
        render_instance(instance, viewport, image, data_buffer, width, height)
    
    return image


def render_wireframe(viewport: Viewport, width, height):
    image = numpy.zeros((height, width, 3))
    for instance in viewport.objects:
        render_wireframe_instance(instance, viewport, image, width, height)
        

    return image
