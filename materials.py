import numpy as np

#——————————[ BLEND FLAGS ]——————————————————————————————————————————————————————————————————————————————————————————————

# Blend flags control how a material's output is composited onto the image.
# Opaque materials are drawn first; alpha, additive and multiply are drawn after, back to front.
BLEND_OPAQUE   = 0  # Overwrites the pixel — only the closest opaque surface is kept
BLEND_ALPHA    = 1  # Blends with the pixel below using the alpha channel: dst = src*a + dst*(1-a)
BLEND_ADDITIVE = 2  # Adds the material colour on top of the pixel: dst = dst + src (useful for glows)
BLEND_MULTIPLY = 3  # Multiplies the material colour with the pixel: dst = dst * src (useful for shadows/tints)

#——————————[ MATERIAL ID ]——————————————————————————————————————————————————————————————————————————————————————————————

def make_id(runtime_id:   int,
            cast_shadow:  bool = True,
            double_sided: bool = False,
            blend_mode:   int  = BLEND_OPAQUE,
            ) -> np.int16:
    """
    Construct a 16-bit material ID from pipeline flags and a runtime-assigned identifier.

    The ID encodes all pipeline-relevant properties of a material into a single integer,
    allowing the renderer to make branching decisions (culling, shadow pass, blend sorting)
    without accessing the material object itself.

    Only flags that cause pipeline branching are stored here. Shader-internal parameters
    (colour, texture, roughness...) live on the material object, not in the ID.

    Bit layout (LSB = bit 0):

        0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
        ╰—┬—╯   │   │   ╰—————————————————————┬——————————————————————╯
        blend   │ double-sided            runtime ID (12 bits)
               cast_shadow

    - bits  0–1 : blend_mode     (2 bits) — BLEND_OPAQUE / ALPHA / ADDITIVE / MULTIPLY
    - bit   2   : cast_shadow    (1 bit)  — whether this material casts shadows
    - bit   3   : double_sided   (1 bit)  — disables back-face culling for this material
    - bits  4–15: runtime_id     (12 bits)— unique material index assigned at runtime (max 4096)

    :param runtime_id:   Unique index assigned by MaterialRegistry at runtime (0–4095)
    :param cast_shadow:  True if this material should appear in the shadow pass
    :param double_sided: True if back-face culling should be disabled (e.g. transparent surfaces)
    :param blend_mode:   One of BLEND_OPAQUE, BLEND_ALPHA, BLEND_ADDITIVE, BLEND_MULTIPLY
    :return: np.int16 encoding all flags
    """
    return np.int16(
        (runtime_id  & 0xFFF) << 4 |
        (double_sided & 0x1)  << 3 |
        (cast_shadow  & 0x1)  << 2 |
        (blend_mode   & 0x3)  << 0
    )

#——————————[ MATERIAL BASE CLASS ]——————————————————————————————————————————————————————————————————————————————————————

class Material:
    """
    Base class for all materials.

    A material defines how a pixel is shaded — it reads from the G-buffer (data_buffer)
    and returns an RGB or RGBA colour for that pixel. Subclasses override get_pixel()
    to implement different shading models (unlit, lit, textured, post-process, etc.).

    The material ID encodes pipeline flags (blend mode, shadow casting, double-sidedness)
    as a 16-bit integer so the renderer can make decisions without touching the material object.
    The runtime portion of the ID is assigned by MaterialRegistry when the material is registered.

    [ FIELDS ]

    id : np.int16 | 16-bit material ID encoding pipeline flags and runtime index (see make_id)

    [ G-BUFFER LAYOUT ]

    The data_buffer is a (height, width, 7) float array written during rasterisation:
        [0:3] — world-space normal, encoded as (n + 1) * 0.5 to fit in [0, 1]
        [3]   — depth as 1/z (larger = closer to camera)
        [4:6] — perspective-correct UV coordinates (u, v)
        [6]   — material ID as float (cast to int before use)

    [ METHODS ]

    get_depth           : Read the 1/z depth of a pixel from the G-buffer
    get_normal          : Read the world-space normal of a pixel from the G-buffer
    get_uv              : Read the UV coordinates of a pixel from the G-buffer
    get_world_pos       : Reconstruct the 3D world position from depth and screen coordinates
    get_pixel           : (override in subclasses) Return the shaded colour for a pixel
    """

    def __init__(self, cast_shadow: bool = False, double_sided: bool = False, blend_mode: int = BLEND_OPAQUE):
        # ID is initialised with runtime_id=0; the real runtime ID is assigned later by MaterialRegistry.register()
        self.id = make_id(0, cast_shadow, double_sided, blend_mode)

    # ——[ G-buffer accessors ]—————————————————————————————————————————————————
    # These helpers abstract the G-buffer layout so subclasses don't hardcode channel indices.

    def get_depth(self, data_buffer: np.ndarray, x: int, y: int) -> np.float64:
        """
        Return the depth (1/z) of pixel (x, y). Larger values are closer to the camera.
        Returns 0 for pixels with no geometry (background).

        :param data_buffer: G-buffer array of shape (height, width, 7)
        :param x: Pixel column
        :param y: Pixel row
        :return: float — 1/z depth value
        """
        return data_buffer[x, y, 3]

    def get_normal(self, data_buffer: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Return the world-space normal of pixel (x, y), decoded from the G-buffer.
        The normal is stored as (n + 1) * 0.5 and must be decoded as n * 2 - 1 to get
        the original [-1, 1] range.

        :param data_buffer: G-buffer array of shape (height, width, 7)
        :param x: Pixel column
        :param y: Pixel row
        :return: np.ndarray of shape (3,) — decoded world-space normal
        """
        encoded = data_buffer[x, y, :3]
        return encoded * 2 - 1  # ← decode from [0,1] back to [-1,1]

    def get_uv(self, data_buffer: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Return the perspective-correct UV coordinates of pixel (x, y).

        :param data_buffer: G-buffer array of shape (height, width, 7)
        :param x: Pixel column
        :param y: Pixel row
        :return: np.ndarray of shape (2,) — (u, v) in [0, 1]
        """
        return data_buffer[x, y, 4:6]

    def get_world_pos(self, data_buffer: np.ndarray, x: int, y: int, camera: object) -> list:
        """
        Reconstruct the 3D world-space position of pixel (x, y) from depth and screen coords.

        This inverts the perspective projection: since projection loses the Z dimension,
        we recover it from 1/z (stored in the G-buffer), then un-project x and y using the
        camera's field of view parameters.

        Requires camera.d (focal length), camera.width, camera.height.

        :param data_buffer: G-buffer array of shape (height, width, 7)
        :param x: Pixel column
        :param y: Pixel row
        :param camera: Camera object with fields d, width, height
        :return: [x_world, y_world, z_world] — 3D position in world/view space
        """
        inv_depth_normalised = data_buffer[x, y, 3]
        if inv_depth_normalised == 0:
            return np.zeros(3)
        # Recover true 1/z from normalised depth
        inv_z = inv_depth_normalised * camera.view_dist  # undo the /view_dist
        z = 1.0 / inv_z
        h, w = data_buffer.shape[:2]
        x_view =  (x - w / 2) * (camera.width  / w) * z / camera.d
        y_view = -(y - h / 2) * (camera.height / h) * z / camera.d
        return [x_view, y_view, z]

    # ——[ Shading ]————————————————————————————————————————————————————————————

    def get_pixel(self, image: np.ndarray, data_buffer: np.ndarray, x: int, y: int, camera: object) -> np.ndarray:
        """
        Return the shaded colour for pixel (x, y). Override in subclasses.

        Opaque materials return RGB (shape (3,)).
        Transparent/post-process materials return RGBA (shape (4,)) where [3] is alpha.

        :param image:       Current image buffer (height, width, 3) — read for blending
        :param data_buffer: G-buffer (height, width, 7) — read for shading inputs
        :param x:           Pixel column
        :param y:           Pixel row
        :param camera:      Camera object
        :return: np.ndarray — RGB or RGBA colour in [0, 1]
        """
        pass

# TODO: Lit shaders
# TODO: Shadows
# TODO: Ambient occlusion (post-process)
# TODO: Fix double-sided and transparency in the rasterizer
# TODO: Group Shader

#——————————[ MATERIALS ]————————————————————————————————————————————————————————————————————————————————————————————————

# UNLIT MATERIALS

class UnlitMaterial(Material):
    """
    Flat-colour material with no lighting. Returns a constant colour for every pixel.
    Useful for debugging, UI elements, or stylised rendering.

    [ FIELDS ]

    color : list | RGB colour in [0, 1]
    """

    def __init__(self, color, cast_shadow=False, double_sided=False):
        super().__init__(cast_shadow, double_sided, BLEND_OPAQUE)
        self.color = color

    def get_pixel(self, image, data_buffer, x, y, camera) -> np.ndarray:
        """Return the flat colour, ignoring all G-buffer data."""
        return self.color


class UnlitTexture(UnlitMaterial):
    """
    Textured unlit material. Samples a texture atlas at the pixel's UV coordinates
    and multiplies the result by a tint colour.

    Inherits from UnlitMaterial so the tint colour can be used as a flat fallback
    if no atlas is provided.

    [ FIELDS ]

    atlas : Atlas | Texture atlas object with a sample(u, v) method returning RGBA
    color : list  | RGB tint multiplied with the texture sample (default white = no tint)
    """

    def __init__(self, atlas, color=[1, 1, 1], aliasing=True, cast_shadow=False, double_sided=False):
        super().__init__(color, cast_shadow, double_sided)
        self.aliasing = aliasing
        self.atlas = atlas

    def get_pixel(self, image, data_buffer, x, y, camera) -> np.ndarray:
        """Sample the atlas at (u, v) and multiply by the tint colour."""
        u, v = self.get_uv(data_buffer, x, y)
        if self.aliasing:
            return np.multiply(self.atlas.sample_bilinear(u, v)[:3], self.color)
        else:
            return np.multiply(self.atlas.sample(u, v)[:3], self.color)

# LIT MATERIALS

class LitMaterial(Material):
    def __init__(self, lights, ambient, diffuse, specular=[1, 1, 1], shininess=100, reflection=0.5, cast_shadow=False, double_sided=False, blend_mode=BLEND_OPAQUE):
        super().__init__(cast_shadow, double_sided, blend_mode)
        self.lights = lights
        self.shininess = shininess
        self.specular = np.array(specular)
        self.diffuse = np.array(diffuse)
        self.ambient = np.array(ambient)
        self.reflection = reflection
        

    def get_pixel(self, image, data_buffer, x, y, camera, clip=True) -> np.ndarray:
        normal = self.get_normal(data_buffer, x, y)
        point = self.get_world_pos(data_buffer, x, y, camera)

        illumination = np.zeros((3))
        for light in self.lights:
            illumination += light.get_illumination(self, camera, point, normal)

        if clip:
            return np.clip(illumination, 0, 1)
        else:
            return illumination
        

class LitTexture (LitMaterial):
    
    def __init__(self, atlas, lights, aliasing=True, ambient=[0.1, 0.1, 0.1], diffuse=[1, 1, 1], specular=[1, 1, 1], shininess=100, reflection=0.5, cast_shadow=False, double_sided=False, blend_mode=BLEND_OPAQUE):
        super().__init__(lights, ambient, diffuse, specular, shininess, reflection, cast_shadow, double_sided, blend_mode)
        self.atlas = atlas
        self.aliasing = aliasing
        
    def get_pixel(self, image, data_buffer, x, y, camera):
        color = super().get_pixel(image, data_buffer, x, y, camera, False)
        u, v = self.get_uv(data_buffer, x, y)
        if self.aliasing:
            return np.clip(np.multiply(self.atlas.sample_bilinear(u, v)[:3], color), 0, 1)
        else:
            return np.clip(np.multiply(self.atlas.sample_bilinear(u, v)[:3], color), 0, 1)
        
#——————————[ POST PROCESS MATERIALS ]———————————————————————————————————————————————————————————————————————————————————

class PostProcess(Material):
    """
    Base class for full-screen post-process effects.

    Post-process materials run after rasterisation on every pixel of the image,
    reading from the G-buffer to produce screen-space effects (fog, bloom, AO, etc.).
    They are not assigned to geometry — they are passed directly to render().

    Unlike regular materials they are never registered in MaterialRegistry.
    """

    def __init__(self, blend_mode):
        super().__init__(False, False, blend_mode)


class PPFog(PostProcess):
    """
    Screen-space distance fog post-process effect.

    Blends a solid fog colour over the image based on each pixel's depth.
    Pixels closer than fog_dist are fully transparent; pixels at fog_dist
    and beyond are fully opaque fog.

    Uses BLEND_ALPHA: fog_colour is composited as dst = src*a + dst*(1-a).

    [ FIELDS ]

    fog_dist  : float     | World-space distance at which fog is fully opaque
    fog_color : list      | RGB colour of the fog in [0, 1]
    """

    def __init__(self, fog_dist, fog_color, blend_mode=BLEND_ALPHA):
        super().__init__(blend_mode)
        self.fog_dist  = fog_dist   # world-space distance at which fog reaches full opacity
        self.fog_color = fog_color

    def get_pixel(self, image, data_buffer, x, y, camera) -> np.ndarray:
        """
        Return the fog colour with alpha proportional to distance.
        Background pixels (depth == 0) are returned as fully opaque fog.
        """
        depth = self.get_depth(data_buffer, x, y)
        if depth == 0:                          # background — no geometry, full fog
            return np.array([*self.fog_color, 1])
        z = 1.0 / depth                         # recover world-space distance from 1/z
        a = 1 - np.clip(self.fog_dist / z, 0, 1)  # 0 = no fog (close), 1 = full fog (far)
        return np.array([*self.fog_color, a])

#——————————[ MATERIAL REGISTRY ]————————————————————————————————————————————————————————————————————————————————————————

class MaterialRegistry:
    """
    Runtime registry that assigns unique IDs to materials.

    The registry deduplicates materials: if two material objects have the same type
    and identical fields, they share the same runtime ID. This avoids wasting ID space
    when the same material is used on multiple instances.

    The runtime ID occupies bits 4–15 of the material ID (12 bits = 4096 max materials).
    The lower 4 bits (blend, shadow, double-sided flags) are preserved from make_id().

    NOTE: __contains__ and _get_id currently iterate over registry keys (IDs) instead
    of values (materials), and call diff_material which is named is_diff_material.
    Both need fixing before the registry works correctly.

    [ FIELDS ]

    _next_id  : int  | Auto-incrementing runtime ID counter (starts at 1)
    _registry : dict | Maps material ID (int) → Material instance

    [ METHODS ]

    register : Assign a runtime ID to a material, deduplicating if already registered
    get      : Retrieve a material by its full 16-bit ID
    """

    _next_id  = 1
    _registry = {}  # { full_16bit_id: Material }

    def __contains__(self, item: Material) -> bool:
        """Return True if an identical material (same type and fields) is already registered."""
        for material in self._registry.values():
            if material == item:
                return True
        return False

    def _get_id(self, item: Material) -> np.int16 | None:
        """Return the ID of an already-registered material identical to item."""
        for material in self._registry.values():
            if material == item:
                return material.id
        return None

    def register(self, material: Material):
        """
        Register a material and assign it a runtime ID.

        If an identical material is already registered, the existing ID is reused.
        Otherwise, a new runtime ID is allocated and the material's id field is updated.

        Raises OverflowError if more than 4096 unique materials are registered.

        :param material: Material instance to register
        """
        if material not in self:
            # Embed the runtime counter into bits 4-15, preserving the lower flag bits
            id = self._next_id << 4 | material.id
            self._next_id += 1
            if (self._next_id << 4) > 0xFFF:
                raise OverflowError("Too many materials (max 4096)")
            self._registry[id] = material
            material.id = id
            # print("new material with id", id, "registered")
        else:
            # Reuse the existing ID for this identical material
            material.id = self._get_id(material)

    def get(self, id: int) -> Material:
        """
        Retrieve a registered material by its full 16-bit ID.

        :param id: Full 16-bit material ID
        :return: Material instance
        """
        return self._registry[id]