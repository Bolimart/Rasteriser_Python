import numpy as np
from objects import *

#——————————[ UTILITIES ]————————————————————————————————————————————————————————————————————————————————————————————————

def normalize(v: np.ndarray) -> np.ndarray:
    """
    Return the unit vector of v.

    :param v: Input vector
    :return:  v divided by its magnitude
    """
    return v / np.linalg.norm(v)

#——————————[ VIEWPORT ]——————————————————————————————————————————————————————————————————————————————————————————————————

class Viewport:
    """
    Holds the scene objects, camera, and projection function used during rendering.

    The viewport is the entry point passed to the renderer — it bundles everything
    the render pipeline needs to produce an image.

    [ FIELDS ]

    objects      : list     | List of Instance objects to render
    camera       : Camera   | Camera defining the view frustum and position
    project_func : callable | Projection function — takes (Triangle, Camera, width, height)
                              and returns a projected Triangle in screen space
    """

    def __init__(self, objects: list, camera: 'Camera', project_func: callable) -> None:
        self.project_func = project_func
        self.camera       = camera
        self.objects      = objects

#——————————[ CAMERA ]————————————————————————————————————————————————————————————————————————————————————————————————————

class Camera:
    """
    A perspective camera defined by a position, a viewing direction, and a virtual
    viewport size.

    The camera coordinate frame is constructed from the viewing normal and an
    arbitrary up vector, producing orthogonal basis vectors u and v that lie on
    the image plane.

    [ FIELDS ]

    pos    : np.ndarray | World-space position of the camera
    normal : np.ndarray | Unit vector pointing in the viewing direction (+Z by default)
    d      : float      | Focal length — distance from the camera to the image plane,
                          controls the field of view (larger = narrower FOV)
    width  : float      | Width of the virtual viewport in world units
    height : float      | Height of the virtual viewport in world units
    u      : np.ndarray | Right basis vector of the image plane
    v      : np.ndarray | Up basis vector of the image plane

    [ METHODS ]

    in_view : Test whether a triangle should be rendered (back-face culling + frustum check)
    """

    def __init__(self,
                 width: float,
                 height: float,
                 d: float,
                 pos: tuple,
                 normal: tuple,
                 view_dist = 1000,
                 gamma = 2.2) -> None:
        self.normal = np.array(normal, dtype=float)
        self.pos    = np.array(pos,    dtype=float)
        self.d      = d
        self.width  = width
        self.height = height
        self.view_dist = view_dist
        self.gamma = gamma

        # Construct an orthonormal basis for the image plane.
        # Pick an arbitrary vector not parallel to the normal, then use cross products.
        arbitrary = np.array([1, 0, 0])
        if abs(np.dot(normal, arbitrary)) > 0.9:  # too parallel — pick a different axis
            arbitrary = np.array([0, 1, 0])
        self.u = normalize(np.cross(arbitrary, normal))  # right vector, lies on the image plane
        self.v = normalize(np.cross(normal, self.u))     # up vector, lies on the image plane

    def in_view(self, triangle: 'Triangle', ignore_back_face: bool = False, debug: bool = False) -> bool:
        """
        Test whether a triangle is visible from this camera.

        Two checks are performed in order:
          1. Back-face culling — the triangle's normal must face toward the camera.
             Skipped when ignore_back_face is True (used for double-sided materials).
          2. Bounding box frustum check — the triangle's screen-space bounding box
             must overlap the viewport.

        The frustum check uses world-space coordinates before projection, so it is
        an approximation. Triangles that straddle the viewport edge are kept.

        :param triangle:         Triangle in world space to test
        :param ignore_back_face: If True, skip back-face culling (double-sided materials)
        :param debug:            If True, print the reason a triangle was rejected
        :return:                 True if the triangle should be rendered, False otherwise
        """
        # Back-face culling — skip triangles whose normal points away from the camera
        to_camera = self.pos - triangle.P0
        if not ignore_back_face and np.dot(triangle.get_normal(), to_camera) <= 0:
            if debug:
                print(f"Rejected — back face (dot={np.dot(triangle.P2_n, to_camera):.3f})")
            return False
        # Todo: add support for smooth shading : Calculate triangle true normal with P0_n P1_n and P2_n

        # Bounding box check — reject if entirely outside the viewport on X or Y
        xs = [triangle.P0[0], triangle.P1[0], triangle.P2[0]]
        ys = [triangle.P0[1], triangle.P1[1], triangle.P2[1]]

        if max(xs) < -self.width / 2 or min(xs) > self.width / 2:
            if debug:
                print("Rejected — outside viewport width")
            return False
        if max(ys) < -self.height / 2 or min(ys) > self.height / 2:
            if debug:
                print("Rejected — outside viewport height")
            return False

        if debug:
            print("Accepted")
        return True