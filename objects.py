import numpy as np
from materials import *

#——————————[ UTILITIES ]————————————————————————————————————————————————————————————————————————————————————————————————

def normalize(v: np.ndarray) -> np.ndarray:
    """
    Return the unit vector of v.

    :param v: Input vector
    :return:  v divided by its magnitude
    """
    return v / np.linalg.norm(v)


def Point2D(x: float, y: float) -> np.ndarray:
    """
    Construct a 2D point as a numpy array.

    :param x: X coordinate
    :param y: Y coordinate
    :return:  np.ndarray of shape (2,)
    """
    return np.array((x, y))


def Point3D(x: float, y: float, z: float) -> np.ndarray:
    """
    Construct a 3D point as a numpy array.

    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return:  np.ndarray of shape (3,)
    """
    return np.array((x, y, z))

def load_obj(obj_path, colour=[1, 1, 1], smooth_shading: bool=True) -> list:
    with open(obj_path, "r") as model:
        lines = []

        # Keep only needed lines
        for line in model:
            # Remove comments
            line = line.strip()
            if '#' in line:
                line = line[:line.index('#')]
            # Ignore empty lines
            if line == "":
                continue
            #Accept only these instruction : v, vn, vt, f, o, and g
            if line[:2] in ['vt', 'vn'] or line[0] in ['o', 'f', 'v', 'g']:
                lines.append(line)

        new_object = lambda: {'vertices': [], 'groups': [], 'faces': []}
        new_group = lambda: {'vertices': [], 'faces': []}

        objects = [new_object()]
        object_num = 0
        group_num = 0
        in_group = False
        vertices = []
        normals = []
        tex_coords = []

        for line in lines:
            if line[0] == 'o':
                object_num += 1
                group_num = 0
                in_group = False
                objects.append(new_object())
            elif line[0] == 'g':
                group_num += 1
                in_group = True
                objects[object_num]['group'].append(new_group())
            elif line[:2] == 'v ':
                point = Point3D(*map(np.float32, line.split()[1:]))
                vertices.append(point)
                if in_group:
                    objects[object_num]['groups'][group_num]['vertices'].append(point)
                else:
                    objects[object_num]['vertices'].append(point)
            elif line[:2] == 'vt':
                point = Point2D(*map(np.float32, line.split()[1:]))
                tex_coords.append(point)
            elif smooth_shading and line[:2] == 'vn':
                point = Point3D(*map(np.float32, line.split()[1:]))
                normals.append(point)
            elif line[0] == 'f':
                vn = []
                v = []
                vt = []
                data = line.split()[1:]
                for p in data:
                    p = p.split('/')
                    v.append(vertices[int(p[0]) - 1])
                    if p[1] != '':
                        vt.append(tex_coords[int(p[1]) - 1])
                    else:
                        vt.append(0)
                    if smooth_shading:
                        vn.append(normals[int(p[2]) - 1])
                if smooth_shading:
                    face = Triangle(v[0], v[1], v[2], colour, vt[0], vt[1], vt[2], vn[0], vn[1], vn[2])
                else:
                    face = Triangle(v[0], v[1], v[2], colour, vt[0], vt[1], vt[2])
                if in_group:
                    objects[object_num]['groups'][group_num]['faces'].append(face)
                else:
                    objects[object_num]['faces'].append(face)
    
    faces_num = 0
    for o in objects:
        faces_num += len(o['faces'])
        for g in o['groups']:
            faces_num += len(g['faces'])
    print(f"New object with {len(vertices)} verticies and {faces_num} faces imported.")
    return objects

#——————————[ PRIMITIVES ]———————————————————————————————————————————————————————————————————————————————————————————————

class Line:
    """
    A line segment defined by two endpoints.

    [ FIELDS ]

    P0 : np.ndarray | Start point
    P1 : np.ndarray | End point
    """

    def __init__(self, P0: np.ndarray, P1: np.ndarray) -> None:
        self.P0 = P0
        self.P1 = P1


class Triangle:
    """
    A triangle defined by three vertices, with an optional colour, normal, and UV coordinates per vertex.

    The normal is computed automatically from the cross product of two edges if not provided.
    Winding order must be counter-clockwise when viewed from outside the mesh for the
    computed normal to point outward correctly.

    [ FIELDS ]

    P0, P1, P2         : np.ndarray | Vertex positions
    P0_uv, P1_uv, P2_uv: list       | Per-vertex UV coordinates in [0, 1]
    V0, V1, V2         : np.ndarray | Edge vectors (P1-P0, P2-P1, P0-P2)
    colour              : list       | RGB colour in [0, 1]
    normal             : np.ndarray | Unit normal vector in world space

    [ METHODS ]

    lines      : Return the three edges of the triangle as Line objects
    get_normal : Compute the unit normal from the cross product of two edges
    """

    def __init__(self,
                 P0: np.ndarray,
                 P1: np.ndarray,
                 P2: np.ndarray,
                 colour: list = [0, 0, 0],
                 P0_uv: list = [0, 0],
                 P1_uv: list = [0, 1],
                 P2_uv: list = [1, 1],
                 P0_n: list = None,
                 P1_n: list = None,
                 P2_n: list = None) -> None:
        # Vertices
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        # Per-vertex UV coordinates
        self.P0_uv = P0_uv
        self.P1_uv = P1_uv
        self.P2_uv = P2_uv
        # Per-vertex normal vectors
        self.P0_n = self.get_normal() if P0_n is None else P0_n
        self.P1_n = self.P0_n if P1_n is None else P1_n
        self.P2_n = self.P0_n if P2_n is None else P2_n
        # Edge vectors — precomputed for reuse in rasterisation and normal calculation
        self.V0 = P1 - P0
        self.V1 = P2 - P1
        self.V2 = P0 - P2

        self.colour = colour

    def lines(self) -> tuple:
        """
        Return the three edges of the triangle as Line objects.

        :return: Tuple of (L0, L1, L2) — edges P0→P1, P1→P2, P2→P0
        """
        return Line(self.P0, self.P1), Line(self.P1, self.P2), Line(self.P2, self.P0)

    def get_normal(self) -> np.ndarray:
        """
        Compute the unit normal of the triangle from the cross product of two edges.

        The direction follows the right-hand rule: counter-clockwise winding (when viewed
        from outside) produces an outward-pointing normal.

        :return: np.ndarray of shape (3,) — unit normal vector
        """
        A = self.P1 - self.P0
        B = self.P2 - self.P0
        return normalize(np.cross(A, B))

#——————————[ MODEL ]————————————————————————————————————————————————————————————————————————————————————————————————————

class Model:
    """
    A 3D mesh defined by a list of vertices and triangles.

    Models are shared across instances — geometry is defined once and transformed
    per-instance at render time. Do not modify a model's triangles at runtime
    if the model is shared between multiple instances.

    [ FIELDS ]

    vertices  : np.ndarray | Array of Point3D vertex positions
    triangles : np.ndarray | Array of Triangle objects referencing those vertices

    [ METHODS ]

    set_colours : Assign colours to a range of triangles
    """

    def __init__(self, verticies=np.array(()), triangles= np.array(())) -> None:
        self.vertices  = verticies
        self.triangles = triangles

    def set_colours(self, colours: list, start: int = 0) -> None:
        """
        Assign a list of colours to triangles starting at a given index.

        :param colours: List of RGB colours in [0, 1], one per triangle
        :param start:  Index of the first triangle to colour (default 0)
        """
        for i, colour in enumerate(colours, start):
            self.triangles[i].colour = colour



#——————————[ TRANSFORM ]————————————————————————————————————————————————————————————————————————————————————————————————

class Transform:
    """
    Stores the position, scale, and rotation of an instance in world space.

    Rotation is stored in degrees and converted to radians during rendering.
    The transform is applied in order: scale → rotate → translate.

    [ FIELDS ]

    pos   : np.ndarray | World-space position (translation)
    scale : np.ndarray | Per-axis scale factors
    rot   : np.ndarray | Euler angles in degrees (rx, ry, rz)
    """

    def __init__(self, pos: np.ndarray, scale: np.ndarray, rot: np.ndarray) -> None:
        self.pos   = pos
        self.scale = scale
        self.rot   = rot

#——————————[ CUBE ]—————————————————————————————————————————————————————————————————————————————————————————————————————

class Cube(Model):
    """
    A unit cube model centered at the origin, spanning [-0.5, 0.5] on all axes.

    Triangles are wound counter-clockwise when viewed from outside, so get_normal()
    produces outward-pointing normals for correct back-face culling.

    Each face covers the full [0, 1] UV range independently, suitable for use with
    a texture atlas where each face maps to a different region.
    """

    def __init__(self) -> None:
        c = [1, 1, 1]
        self.vertices = np.array((
            Point3D(-0.5, -0.5, -0.5),  # 0
            Point3D( 0.5, -0.5, -0.5),  # 1
            Point3D( 0.5,  0.5, -0.5),  # 2
            Point3D(-0.5,  0.5, -0.5),  # 3
            Point3D(-0.5, -0.5,  0.5),  # 4
            Point3D( 0.5, -0.5,  0.5),  # 5
            Point3D( 0.5,  0.5,  0.5),  # 6
            Point3D(-0.5,  0.5,  0.5),  # 7
        ))
        v = self.vertices
        self.triangles = np.array((
            # Front face (z-) — normal points toward -Z
            Triangle(v[0], v[2], v[1], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], colour=c),
            Triangle(v[0], v[3], v[2], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], colour=c),
            # Back face (z+) — normal points toward +Z
            Triangle(v[4], v[5], v[7], P0_uv=[0,0], P1_uv=[1,0], P2_uv=[0,1], colour=c),
            Triangle(v[7], v[5], v[6], P0_uv=[0,1], P1_uv=[1,0], P2_uv=[1,1], colour=c),
            # Left face (x-) — normal points toward -X
            Triangle(v[0], v[4], v[3], P0_uv=[0,0], P1_uv=[1,0], P2_uv=[0,1], colour=c),
            Triangle(v[7], v[3], v[4], P0_uv=[1,1], P1_uv=[0,1], P2_uv=[1,0], colour=c),
            # Right face (x+) — normal points toward +X
            Triangle(v[1], v[6], v[5], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], colour=c),
            Triangle(v[1], v[2], v[6], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], colour=c),
            # Bottom face (y-) — normal points toward -Y
            Triangle(v[4], v[1], v[5], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], colour=c),
            Triangle(v[4], v[0], v[1], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], colour=c),
            # Top face (y+) — normal points toward +Y
            Triangle(v[3], v[6], v[2], P0_uv=[0,0], P1_uv=[1,1], P2_uv=[1,0], colour=c),
            Triangle(v[3], v[7], v[6], P0_uv=[0,0], P1_uv=[0,1], P2_uv=[1,1], colour=c),
        ))

#——————————[ INSTANCE ]—————————————————————————————————————————————————————————————————————————————————————————————————

def make_instance(model: list[Model], transform: Transform,  mat: Material|list[Material]):
    if model is Model:
        return Instance(model, transform, mat)
    elif len(model) == 1:
        return Instance(model[0], transform, mat[0])
    else:
        return ComplexInstance(model, transform, mat)

# TODO: add bounding box for frustrum culling
class Instance:
    """
    A positioned, scaled, and rotated occurrence of a Model in the scene.

    Multiple instances can share the same Model, each with its own Transform and Material.
    The transform is applied in order: scale → rotate (Rz @ Ry @ Rx) → translate.

    [ FIELDS ]

    model     : Model     | Shared mesh geometry
    transform : Transform | Position, scale, and rotation in world space
    mat       : Material  | Material used to shade this instance

    [ METHODS ]

    apply_transform : Return all triangles of the model transformed to world space
    scale           : Apply per-axis scaling to a triangle
    rotate          : Apply Euler rotation (XYZ order) to a triangle
    translate       : Apply world-space translation to a triangle
    """

    def __init__(self, model: Model, transform: Transform, material: Material = Material()) -> None:
        self.model     = model
        self.transform = transform
        self.mat       = material

    def apply_transform(self) -> list:
        """
        Transform all triangles of the model to world space.

        Applies scale, rotation, and translation in that order.

        :return: List of Triangle objects in world space
        """
        transformed = []
        for t in self.model.triangles:
            scaled     = self.scale(t)
            rotated    = self.rotate(scaled)
            translated = self.translate(rotated)
            transformed.append(translated)
        return transformed

    def scale(self, t: Triangle) -> Triangle:
        """
        Scale a triangle's vertices by the instance's per-axis scale factors.

        :param t: Input triangle in local space
        :return:  Triangle with scaled vertices
        """
            # Scale vertices
        P0 = t.P0 * self.transform.scale
        P1 = t.P1 * self.transform.scale
        P2 = t.P2 * self.transform.scale

        # Scale normals by the inverse of the scale factors
        inv_scale = 1.0 / self.transform.scale
        P0_n = t.P0_n * inv_scale
        P1_n = t.P1_n * inv_scale
        P2_n = t.P2_n * inv_scale

        # Renormalize the normals to ensure they are unit vectors
        P0_n = normalize(P0_n)
        P1_n = normalize(P1_n)
        P2_n = normalize(P2_n)

        return Triangle(
            P0, P1, P2,
            t.colour,
            t.P0_uv, t.P1_uv, t.P2_uv,
            P0_n, P1_n, P2_n
        )

    def rotate(self, t: Triangle) -> Triangle:
        """
        Rotate a triangle's vertices using Euler angles (X → Y → Z order).

        Rotation matrices are constructed from the instance's rot field (in degrees)
        and combined as R = Rz @ Ry @ Rx, then applied to each vertex.

        :param t: Input triangle
        :return:  Triangle with rotated vertices
        """
        def rotate_vertex(v: np.ndarray) -> np.ndarray:
            rx, ry, rz = np.radians(self.transform.rot)

            Rx = np.array([
                [1,           0,            0],
                [0,  np.cos(rx), -np.sin(rx)],
                [0,  np.sin(rx),  np.cos(rx)],
            ])
            Ry = np.array([
                [ np.cos(ry), 0, np.sin(ry)],
                [0,           1,          0],
                [-np.sin(ry), 0, np.cos(ry)],
            ])
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz),  np.cos(rz), 0],
                [0,           0,          1],
            ])

            R = Rz @ Ry @ Rx  # Combined rotation: X applied first, then Y, then Z
            return R @ np.array(v)

        # Rotate vertices
        P0 = rotate_vertex(t.P0)
        P1 = rotate_vertex(t.P1)
        P2 = rotate_vertex(t.P2)

        # Rotate normals using the same rotation matrix
        P0_n = rotate_vertex(t.P0_n)
        P1_n = rotate_vertex(t.P1_n)
        P2_n = rotate_vertex(t.P2_n)

        # Renormalize the normals to ensure they are unit vectors
        P0_n = normalize(P0_n)
        P1_n = normalize(P1_n)
        P2_n = normalize(P2_n)

        return Triangle(
            P0, P1, P2,
            t.colour,
            t.P0_uv, t.P1_uv, t.P2_uv,
            P0_n, P1_n, P2_n
        )

    def translate(self, t: Triangle) -> Triangle:
        """
        Translate a triangle's vertices by the instance's world-space position.

        :param t: Input triangle
        :return:  Triangle with translated vertices
        """
        P0 = t.P0 + self.transform.pos
        P1 = t.P1 + self.transform.pos
        P2 = t.P2 + self.transform.pos
        return Triangle(P0, P1, P2, t.colour, t.P0_uv, t.P1_uv, t.P2_uv, t.P0_n, t.P1_n, t.P2_n)


class ComplexInstance(Instance):
    """
        A positioned, scaled, and rotated occurrence of multiple Model in the scene, with the same transform.

        Multiple instances can share the same Model, each with its own Transform and Material.
        The transform is applied in order: scale → rotate (Rz @ Ry @ Rx) → translate.

        [ FIELDS ]

        models     : list[Model]     | Shared mesh geometry
        transform  : Transform       | Position, scale, and rotation in world space
        mats       : list[Material]  | Material used to shade this instance
        instances  : list[Instances] | Each model has a unique instance

        [ METHODS ]

        apply_transform : Return all triangles of the models transformed to world space
        scale           : Apply per-axis scaling to a triangle
        rotate          : Apply Euler rotation (XYZ order) to a triangle
        translate       : Apply world-space translation to a triangle
        """

    def __init__(self, models: list[Model], transform: Transform, mats: list[Material]):
        to_model = lambda model: Model(model['vertices'], model['faces'])
        if type(mats) is not list:
            self.instances = [Instance(to_model(models[i]), transform, mats) for i in range(len(models))]
        else:
            self.instances = [Instance(to_model(models[i]), transform, mats[min(i, len(mats) - 1)]) for i in range(len(models))]


    def apply_transform(self) -> list:
        transformed = []
        for instance in self.instances:
            transformed += instance.apply_transform()
        return transformed

#——————————[ ATLAS ]————————————————————————————————————————————————————————————————————————————————————————————————————

class Atlas:
    """
    A texture atlas that supports UV sampling with automatic uint8 → float conversion.

    UV coordinates follow the convention that (0, 0) is the bottom-left corner
    (V is flipped relative to image row order, where row 0 is the top).

    [ FIELDS ]

    texture : np.ndarray | Image array of shape (height, width, 3) or (height, width, 4),
                           either float [0, 1] or uint8 [0, 255]

    [ METHODS ]

    sample : Return the colour at a given (u, v) coordinate
    """

    def __init__(self, texture: np.ndarray) -> None:
        self.texture = texture

    def sample(self, u: float, v: float) -> np.ndarray:
        """
        Sample the texture at UV coordinates (u, v).

        UV coordinates are clamped to [0, 1]. V is flipped so that (0, 0) maps
        to the bottom-left of the image. uint8 textures are normalised to [0, 1].

        :param u: Horizontal texture coordinate in [0, 1]
        :param v: Vertical texture coordinate in [0, 1], (0, 0) = bottom-left
        :return:  np.ndarray — RGBA or RGB colour in [0, 1]
        """
        h, w = self.texture.shape[:2]
        u = min(max(u, 0), 1)
        v = min(max(v, 0), 1)
        tx = int(u * (w - 1))
        ty = int((1 - v) * (h - 1))  # flip V: row 0 is top, but UV (0,0) is bottom-left
        if self.texture.dtype == np.uint8 :
            return np.clip(self.texture[ty, tx] / 255.0, 0, 1)
        return self.texture[ty, tx]

    def sample_bilinear(self, u: float, v: float) -> np.ndarray:
        h, w = self.texture.shape[:2]
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)

        # Continuous texel coordinates
        tx = u * (w - 1)
        ty = (1 - v) * (h - 1)

        # Four surrounding texel indices
        x0, y0 = int(tx), int(ty)
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        # Fractional parts (how far between the two texels)
        fx = tx - x0
        fy = ty - y0

        # Fetch the four texels
        def fetch(y, x):
            c = self.texture[y, x].astype(np.float64)
            if self.texture.dtype == np.uint8:
                c /= 255.0
            return c

        c00 = fetch(y0, x0)  # top-left
        c10 = fetch(y0, x1)  # top-right
        c01 = fetch(y1, x0)  # bottom-left
        c11 = fetch(y1, x1)  # bottom-right

        # Bilinear blend
        top = c00 * (1 - fx) + c10 * fx
        bottom = c01 * (1 - fx) + c11 * fx
        return np.clip(top * (1 - fy) + bottom * fy, 0, 1)