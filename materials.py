import inspect
import numpy as np

#---------[ MATERIAL CLASS ]------------------------------------------------------------------

BLEND_OPAQUE = 0
BLEND_ALPHA = 1
BLEND_ADDITIVE = 2
BLEND_MULTIPLY = 3

def diff_material(mat1, mat2):
    attrs1 = {k: v for k, v in inspect.getmembers(mat1) if not k.startswith('__') and not inspect.ismethod(v)}
    attrs2 = {k: v for k, v in inspect.getmembers(mat2) if not k.startswith('__') and not inspect.ismethod(v)}
    diffs = {}
    for k in set(attrs1.keys).union(attrs2.keys()):
        if attrs1.get(k) != attrs2.get(k):
            diffs[k] = (attrs1.get(k), attrs2.get(k))
    print("Diff found : ", diffs)
    return diffs

# TODO: Fix the comparaison


class Material:
    
    def __init__(self, cast_shadow=False, double_sided=False, blend_mode=BLEND_OPAQUE):
        # Flags
        self.id = self.make_id(0, cast_shadow, double_sided, blend_mode)
        
    def make_id(self,
        arbitrary,                      # 12bit : There can be 4096 differents shader !
        cast_shadow = True,             # 1bit
        double_sided = False,           # 1bit
        blend_mode = BLEND_OPAQUE,      # 2bit (0=opaque, 1=alpha, 2=additive, 3=multiply) or use flags
        ):
        # The Id is composed from multiple flags and an arbitrary number, different from each material.
        # Each material as a lenght of 16bits, so you can have 2 material ID stocked in the same 32bit int or 4 in a 64bit int
        # allowing for transparency
        # At the start, the arbitrary number is 0, and is set at runtime
        return  ((arbitrary    & 0xFFF)  << 4 |
                 (double_sided  & 0x1)   << 3 |
                 (cast_shadow   & 0x1)   << 2 |
                 (blend_mode    & 0x3)   << 0 )
        
    # Utilities for the get pixel function
    def get_depth(self, data_buffer, x, y):
        return data_buffer[x, y, 3]
    
    def get_normal(self, data_buffer, x, y):
        return data_buffer[x, y, :3]
    
    def get_uv(self, data_buffer, x, y):
        return data_buffer[x, y, 4:6]
    
    def get_data_buffer_dimensions(self, data_buffer):
        return len(data_buffer), len(data_buffer[0])
    
    def get_world_coordinates(self, data_buffer, x, y, camera):
        # In your shader, given pixel screen coords (x, y) and depth d = 1/z
        z = 1.0 / self.get_depth(data_buffer, x, y)

        d = self.get_data_buffer_dimensions(data_buffer)
        # Reverse the viewport_to_canvas transform to get view-space x, y
        x_view = (x - d[0]/2) * (camera.width / d[0]) * z / camera.d
        y_view = -(y - d[1]/2) * (camera.height / d[1]) * z / camera.d

        return [x_view, y_view, z]
    
    def get_pixel(self, image, data_buffer, x, y, camera):
        pass
    
# TODO: Post process: Ambiant Oclusion
# TODO: shadow ?
# TODO: double sided, Transparency
# TODO: Lit shaders

#---------[ MATERIALS ]--------------------------------------------------------------------------------

class Unlit_Material(Material):
    
    def __init__(self, color, cast_shadow=False, double_sided=False):
        super().__init__(cast_shadow, double_sided, BLEND_OPAQUE)
        self.color = color
        
    def get_pixel(self, image, data_buffer, x, y, camera):
        return self.color
    

class Unlit_Texture (Unlit_Material):
    
    def __init__(self, atlas, color=[1, 1, 1], cast_shadow=False, double_sided=False):
        super().__init__(color, cast_shadow, double_sided)
        self.atlas = atlas
        
    def get_pixel(self, image, data_buffer, x, y, camera):
        u, v = self.get_uv(data_buffer, x, y)
        return np.multiply(self.atlas.sample(u, v), self.color)

#---------[ POST PROCESS MATERIALS ]------------------------------------------------------------------

class PostProcess (Material):
    
    def __init__(self, blend_mode):
        super().__init__(False, False, blend_mode)

class PPFog (PostProcess):
    
    def __init__(self, fog_factor, fog_color, blend_mode=BLEND_ALPHA):
        super().__init__(blend_mode)
        self.fog_factor = fog_factor
        self.fog_color = fog_color
        
    def get_pixel(self, image, data_buffer, x, y, camera):
        depth = self.get_depth(data_buffer, x, y)
        if depth == 0:  # no geometry here, skip
            return np.array([*self.fog_color, 1])
        z = depth
        a = np.clip(1 - z * self.fog_factor, 0, 1)
        return np.array([*self.fog_color, a])
    
#---------[ MATERIAL REGISTRY ]------------------------------------------------------------------
    
class MaterialRegistry:
    _next_id = 1
    _registry = {}
    
    def __contains__(self, item):
        # Check if a material with the exact same fields is already in it
        for material in self._registry:
            if type(material) == type(item) and diff_material(material, item):
                return True
        return False
    
    def _get_id(self, item):
        for material in self._registry:
            if type(material) == type(item) and diff_material(material, item):
                return material
    
    def register(self, material):
        if material not in self:
            id = self._next_id << 4 | material.id
            self._next_id += 1
            if (self._next_id << 4) > 0xFFF:
                raise OverflowError("Too many materials (max 4096)")
            self._registry[id] = material
            material.id = id
        else: 
            material.id = self._get_id(material)
                
    def get(self, id):
        return self._registry[id]
