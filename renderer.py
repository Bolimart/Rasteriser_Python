from math import floor

import numpy
import numpy as np
from objects import *
from viewport import *


# =====[ Utilities ]=====

def swap (x0, x1):
    return x1, x0

def lerp(a, b, t):
    return a + t * (b - a)

def interp(i0, d0, i1, d1):
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

def interp_float(i0, d0, i1, d1):
    """Float interpolation — for depth and normals"""
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

# =====[ Draw Functions ]=====


def draw_triangle(triangle, data_buffer,  view_dist, mat_id):
    
    # This one is way more optimized, but less straight forward
    x0, y0, depth0 = triangle.P0
    x1, y1, depth1 = triangle.P1
    x2, y2, depth2 = triangle.P2
    u0, v0 = triangle.P0_uv
    u1, v1 = triangle.P1_uv
    u2, v2 = triangle.P2_uv
    
    # 1 - Sort the points so that y0 <= y1 <= y2
    if y1 < y0: x0,x1=swap(x0,x1); y0,y1=swap(y0,y1); depth0,depth1=swap(depth0,depth1); u0,u1=swap(u0,u1); v0,v1=swap(v0,v1)
    if y2 < y0: x0,x2=swap(x0,x2); y0,y2=swap(y0,y2); depth0,depth2=swap(depth0,depth2); u0,u2=swap(u0,u2); v0,v2=swap(v0,v2)
    if y2 < y1: x1,x2=swap(x1,x2); y1,y2=swap(y1,y2); depth1,depth2=swap(depth1,depth2); u1,u2=swap(u1,u2); v1,v2=swap(v1,v2)
    # The "tall" side (the side with the > Y0 - Y1 is always P0 to P2

    # 2 - Compute the x coordinates of the triangle edges
    x01 = interp(y0, x0, y1, x1); x12 = interp(y1, x1, y2, x2); x02 = interp(y0, x0, y2, x2)
    u01 = interp_float(y0, u0, y1, u1); u12 = interp_float(y1, u1, y2, u2); u02 = interp_float(y0, u0, y2, u2)
    v01 = interp_float(y0, v0, y1, v1); v12 = interp_float(y1, v1, y2, v2); v02 = interp_float(y0, v0, y2, v2)
    d01 = interp_float(y0, depth0, y1, depth1); d12 = interp_float(y1, depth1, y2, depth2); d02 = interp_float(y0, depth0, y2, depth2)

    # 3 - Concatenate the short sides
    x01.pop() # Remove the last coordinate of x01 as it is the same as the first of x12
    x012 = x01 + x12; del(x01, x12)
    d01.pop(); d012 = d01 + d12; del(d01, d12)
    u01.pop(); u012 = u01 + u12; del(u01, u12)
    v01.pop(); v012 = v01 + v12; del(v01, v12)

    # 4 - Determine which is left and which is right
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
    del(x02, x012); del(d02, d012); del(u02, u012); del(v02, v012)

    # 5 - Draw the horizontal segments
    height, width = data_buffer.shape[:2]
    y0, y2 = int(y0), int(y2)
    
    for y in range(y0, y2):
        i = y - y0
        xl, xr = int(x_left[i]), int(x_right[i])
        dl, dr = d_left[i], d_right[i]
        ul, ur = u_left[i], u_right[i]
        vl, vr = v_left[i], v_right[i]
        
        for x in range(xl, xr):
            
            # Don't draw if the pixel is out of the screen
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Interpolate depth across the scanline
            t = (x - xl) / (xr - xl) if xr != xl else 0
            d = float(lerp(dl, dr, t))
            inv_depth = min(max(d / view_dist, 0), 1) if d != 0 else 0

            # interpolate the uv coordinates
            u = lerp(ul, ur, t)
            v = lerp(vl, vr, t)
            
            if data_buffer[y, x, 3] >= inv_depth:
                continue
            
            normal = [
                (triangle.normal[0] + 1) * 0.5,
                (triangle.normal[1] + 1) * 0.5,
                (triangle.normal[2] + 1) * 0.5                
            ]
            
            data_buffer[y, x, 0:3] = normal  # world-space normal (xyz)
            data_buffer[y, x, 3] = inv_depth                            # 1/z
            data_buffer[y, x, 4:6] = [u, v]                             # uv coordinates
            data_buffer[y, x, 6] = mat_id                               # material ID                
            


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
        ys = interp(x0, y0, x1, y1)
        for x in range(x0, x1 + 1):
            if 0 <= x < w and 0 <= ys[x - x0] < h:
                image[ys[x - x0], x] = color

    # The line is vertical-ish
    else:
        if y0 > y1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)
        xs = interp(y0, x0, y1, x1)
        for y in range(y0, y1 + 1):
            if 0 <= y < h and 0 <= xs[y - y0] < w:
                image[y, xs[y - y0]] = color
                

def draw_pixel(material, image, data_buffer, x, y, camera):
    c = np.zeros(4)
    blend = (material.id >> 0) & 0x3
    if blend == BLEND_OPAQUE:
        image[x, y] = material.get_pixel(image, data_buffer, x, y, camera) # Get the material color
    elif blend == BLEND_ALPHA:
        c = np.array(material.get_pixel(image, data_buffer, x, y, camera), dtype=float)
        image[x, y] = image[x, y] * (1 - c[3]) + c[:3] * c[3]
    elif blend == BLEND_ADDITIVE:
        c[:3] = material.get_pixel(image, data_buffer, x, y, camera)
        print(c)
        image[x, y] = np.clip(image[x, y] + c[:3], 0, 1)
    elif blend == BLEND_MULTIPLY:
        c[:3] = material.get_pixel(image, data_buffer, x, y, camera)
        image[x, y] = np.clip(image[x, y] * c[:3], 0, 1) 
                        
    


# =====[ Projections ]=====

def perspective_projection(triangle: Triangle, camera: Camera, width, height):
    def viewport_to_canvas(x, y, z):
        cx = width / 2 + x * (width / camera.width)
        cy = height / 2 - y * (height / camera.height)  # flip Y so +Y is up
        return Point3D(cx, cy, 1.0 / z)
    
    def project_vertex(v):
        P0 = v[0] * (camera.d / v[2])
        P1 = v[1] * (camera.d / v[2])
        return viewport_to_canvas(P0 , P1, v[2])

    P0 = project_vertex(triangle.P0)
    P1 = project_vertex(triangle.P1)
    P2 = project_vertex(triangle.P2)
    if abs(P0 - P1).any() < 10e-3 and abs(P2 - P1).any() < 10e-3 and abs(P0 - P2).any() < 10e-3 : # Don't draw small enough triangles
            return None
    
    return Triangle(P0, P1, P2, triangle.color, triangle.normal, triangle.P0_uv, triangle.P1_uv, triangle.P2_uv)

# =====[ Render ]=====

def render_wireframe_instance(instance:Instance, viewport: Viewport, image, width, height):
    projected = []
    for t in instance.apply_transform():
        projected.append(viewport.project_func(t, viewport.camera, width, height))
    for T in projected:
        draw_wireframe_triangle(T, image)
    

def render_instance(instance:Instance, viewport: Viewport, data_buffer,  width, height, view_dist):
    projected = []
    for t in instance.apply_transform():
        if viewport.camera.in_view(t):
            T = viewport.project_func(t, viewport.camera, width, height)
            if T is not None:
                projected.append(T)
    for T in projected:
        draw_triangle(T, data_buffer,  view_dist, instance.mat.id)


def render(viewport:Viewport, width, height, view_dist, post_process=[]):

    materials = MaterialRegistry()
    image = numpy.zeros((height, width, 3))
    data_buffer = numpy.zeros((height, width, 7))

    for instance in viewport.objects:
        materials.register(instance.mat)
        render_instance(instance, viewport, data_buffer, width, height, view_dist)
    
    for x in range(len(data_buffer)):
        for y in range(len(data_buffer[0])):
            material_data = int(data_buffer[x, y, 6])
            if material_data != 0:
                for i in range(material_data.bit_length()//16 + 1):
                    material_id = material_data >> 16 * i
                    material = materials.get(material_id)
                    draw_pixel(material, image, data_buffer, x, y, viewport.camera)
    
    for x in range(len(data_buffer)):
        for y in range(len(data_buffer[0])):
            for pp in post_process:
                draw_pixel(pp, image, data_buffer, x, y, viewport.camera)
    
    return image, data_buffer

# TODO: Camera Rotation     

def render_wireframe(viewport: Viewport, width, height):
    image = numpy.zeros((height, width, 3))
    for instance in viewport.objects:
        render_wireframe_instance(instance, viewport, image, width, height)
        

    return image
