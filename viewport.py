import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)

class Viewport:

    def __init__(self, objects, camera, project_func):
        self.project_func = project_func
        self.camera = camera
        self.triangles = []
        for obj in objects:
            self.triangles += obj.triangles


    def project(self, width, height):
        triangles2D = []
        for triangle in self.triangles:
            triangles2D.append(self.project_func(triangle, self.camera, width, height))

        return triangles2D


class Camera:

    def __init__(self, width, height, d, pos, normal):
        self.normal = np.array(normal, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.d = d
        self.width = width
        self.height = height
        # Pick an arbitrary vector not orthogonal to the normal
        arbitrary = np.array([1, 0, 0])
        if abs(np.dot(normal, arbitrary)) > 0.9:  # too orthogonal, pick another
            arbitrary = np.array([0, 1, 0])
        self.u = normalize(np.cross(arbitrary, normal)) # Lies on the plane
        self.v = normalize(np.cross(normal, self.u))
