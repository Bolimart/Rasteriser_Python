import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def left_normal(v):
    return normalize(np.array([-v[1], v[0]]))


def right_normal(v):
    return normalize(np.array([v[1], -v[0]]))


def Point2D(x, y):
    return np.array((x, y))


def Point3D(x, y, z):
    return np.array((x, y, z))


def rotation_matrix(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-10:
        return np.eye(3)  # no rotation
    axis = rotation_vector / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ]
    ])


class Line:
    def __init__(self, P0, P1):
        self.P0 = P0
        self.P1 = P1


class Triangle:
    
    def __init__(self, P0, P1, P2, color):
        # Store the points
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        # Store the vectors
        self.V0 = P1 - P0
        self.V1 = P2 - P1
        self.V2 = P0 - P2

        self.color = color
        
    def is_point_inside(self, P):
        # Use the dot product to determine whether a point is inside the triangle
        P0_to_P = P - self.P0
        P1_to_P = P - self.P1
        P2_to_P = P - self.P2
        N0 = left_normal(self.V0)
        N1 = left_normal(self.V1)
        N2 = left_normal(self.V2)

        if np.dot(P0_to_P, N0) < 0 and np.dot(P1_to_P, N1) < 0 and np.dot(P2_to_P, N2) < 0:
            return True
        else:
            return False

    def lines(self):
        # Store the lines
        L0 = Line(self.P0, self.P1)
        L1 = Line(self.P1, self.P2)
        L2 = Line(self.P2, self.P0)
        return L0, L1, L2

class Model:

    def __init__(self):
        self.vertices = np.array(())
        self.triangles = np.array(())
        
    def set_colors(self, colors, start=0):
        for i, color in enumerate(colors, start):
            self.triangles[i].color = color



class Cube(Model):

    def __init__(self):
        c = [1, 1, 1]
        self.vertices = np.array((
            Point3D(0, 0, 0),
            Point3D(1, 0, 0),
            Point3D(1, 1, 0),
            Point3D(0, 1, 0),
            Point3D(0, 0, 1),
            Point3D(1, 0, 1),
            Point3D(1, 1, 1),
            Point3D(0, 1, 1),
        ))
        v = self.vertices
        self.triangles = np.array((
            # Front face  (z)
            Triangle(v[0], v[1], v[2], c),
            Triangle(v[0], v[2], v[3], c),
            # Back face   (z +c
            Triangle(v[5], v[4], v[7], c),
            Triangle(v[5], v[7], v[6], c),
            # Left face   (x)
            Triangle(v[4], v[0], v[3], c),
            Triangle(v[4], v[3], v[7], c),
            # Right face  (x +c
            Triangle(v[1], v[5], v[6], c),
            Triangle(v[1], v[6], v[2], c),
            # Bottom face (y)
            Triangle(v[4], v[5], v[1], c),
            Triangle(v[4], v[1], v[0], c),
            # Top face    (y +c
            Triangle(v[3], v[2], v[6], c),
            Triangle(v[3], v[6], v[7], c)
        ))


class Instance:

    def __init__(self, model, pos, scale, rotation):
        self.model = model
        self.pos = pos
        self.scale = scale
        self.rotation = rotation
