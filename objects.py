import numpy as np
from materials import Material


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


class Line:
    def __init__(self, P0, P1):
        self.P0 = P0
        self.P1 = P1


class Triangle:
    
    def __init__(self, P0, P1, P2, color=[0, 0, 0], normal=None, P0_uv=[0, 0], P1_uv=[0, 1], P2_uv=[1, 1]):
        # Store the points
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        # Store the uv coordinates of each point
        self.P0_uv = P0_uv
        self.P1_uv = P1_uv
        self.P2_uv = P2_uv
        # Store the vectors
        self.V0 = P1 - P0
        self.V1 = P2 - P1
        self.V2 = P0 - P2

        self.color = color
        if normal is None:
            self.normal = self.get_normal()
        else:
            self.normal = normal
        
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
    
    def get_normal(self):
        # Two edges of the triangle
        A = np.array(self.P1) - np.array(self.P0)
        B = np.array(self.P2) - np.array(self.P0)
        
        return normalize(np.cross(A, B))

class Model:

    def __init__(self):
        self.vertices = np.array(())
        self.triangles = np.array(())
        
    def set_colors(self, colors, start=0):
        for i, color in enumerate(colors, start):
            self.triangles[i].color = color


class Transform:
    
    def __init__(self, pos, scale, rot):
        self.pos = pos
        self.scale = scale
        self.rot = rot


class Cube(Model):

    def __init__(self):
        c = [1, 1, 1]
        self.vertices = np.array((
            Point3D(-0.5, -0.5, -0.5),
            Point3D(0.5, -0.5, -0.5),
            Point3D(0.5, 0.5, -0.5),
            Point3D(-0.5, 0.5, -0.5),
            Point3D(-0.5, -0.5, 0.5),
            Point3D(0.5, -0.5, 0.5),
            Point3D(0.5, 0.5, 0.5),
            Point3D(-0.5, 0.5, 0.5),
        ))
        v = self.vertices
        self.triangles = np.array((
    # Front face (z-)
    Triangle(v[0], v[2], v[1], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], color=c),
    Triangle(v[0], v[3], v[2], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], color=c),
    # Back face (z+)
    Triangle(v[4], v[5], v[7], P0_uv=[0,0], P1_uv=[1,0], P2_uv=[0,1], color=c),
    Triangle(v[7], v[5], v[6], P0_uv=[0,1], P1_uv=[1,0], P2_uv=[1,1], color=c),
    # Left face (x-)
    Triangle(v[0], v[4], v[3], P0_uv=[0,0], P1_uv=[1,0], P2_uv=[0,1], color=c),
    Triangle(v[7], v[3], v[4], P0_uv=[1,1], P1_uv=[0,1], P2_uv=[1,0], color=c),
    # Right face (x+)
    Triangle(v[1], v[6], v[5], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], color=c),
    Triangle(v[1], v[2], v[6], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], color=c),
    # Bottom face (y-)
    Triangle(v[4], v[1], v[5], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], color=c),
    Triangle(v[4], v[0], v[1], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], color=c),
    # Top face (y+)
    Triangle(v[3], v[6], v[2], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], color=c),
    Triangle(v[3], v[7], v[6], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], color=c),
))


class Instance:

    def __init__(self, model: Model, transform: Transform, materials: list[Material] = None):
        self.model = model
        self.transform = transform
        self.materials = materials
        
        
    def get_pixel_color(depth, normal, uv_coord, screen_coord):
        pass
        
    
    def apply_transform(self):
        transformed = []
        for t in self.model.triangles:
            scaled = self.scale(t)
            rotated = self.rotate(scaled)
            translated = self.translate(rotated)
            transformed.append(translated)
        return transformed
            
            
    def scale(self, t):
        P0 = (t.P0 * self.transform.scale)
        P1 = (t.P1 * self.transform.scale)
        P2 = (t.P2 * self.transform.scale)
        return Triangle(P0, P1, P2, t.color, P0_uv=t.P0_uv, P1_uv=t.P1_uv, P2_uv=t.P2_uv)
       
    def rotate(self, t):
        def rotate_vertex(v):
            rx, ry, rz = np.radians(self.transform.rot)  # convert degrees → radians

            # Rotation around X
            Rx = np.array([
                [1,           0,            0],
                [0,  np.cos(rx), -np.sin(rx)],
                [0,  np.sin(rx),  np.cos(rx)],
            ])
            # Rotation around Y
            Ry = np.array([
                [ np.cos(ry), 0, np.sin(ry)],
                [0,           1,          0],
                [-np.sin(ry), 0, np.cos(ry)],
            ])
            # Rotation around Z
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz),  np.cos(rz), 0],
                [0,           0,          1],
            ])

            R = Rz @ Ry @ Rx        # combined rotation matrix
            return R @ np.array(v)  # apply to vertex
        P0 = rotate_vertex(t.P0)
        P1 = rotate_vertex(t.P1)
        P2 = rotate_vertex(t.P2)
        return Triangle(P0, P1, P2, t.color, P0_uv=t.P0_uv, P1_uv=t.P1_uv, P2_uv=t.P2_uv)
    
    def translate(self, t):
        P0 = t.P0 + self.transform.pos
        P1 = t.P1 + self.transform.pos
        P2 = t.P2 + self.transform.pos
        return Triangle(P0, P1, P2, t.color, P0_uv=t.P0_uv, P1_uv=t.P1_uv, P2_uv=t.P2_uv)
        