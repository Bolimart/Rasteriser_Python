import numpy as np

def Point2D(x, y):
    return np.array((x, y))

def Point3D(x, y, z):
    return np.array((x, y, z))

class Line:
    
    def __init__(self, P0, P1):
        self.P0 = P0
        self.P1 = P1
        
class triangle:
    
    def __init__(self, P0, P1, P2):
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        self.L0 = Line(P0, P1)
        self.L1 = Line(P1, P2)
        self.L2 = Line(P2, P0)
        
    def is_point_inside(self, P):