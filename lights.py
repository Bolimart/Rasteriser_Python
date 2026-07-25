import numpy as np
from objects import normalize

# ——————————[ LIGHTS ]———————————————————————————————————————————————————————————————————————————————————————————————————

class PointLight:
    """
    A point light source with a position, colour, and intensity.

    [ FIELDS ]

    colour     : list  | RGB colour of the light in [0, 1]
    pos       : np.ndarray | World-space position of the light
    intensity : float | Brightness multiplier
    """

    def __init__(self, pos, ambient=np.array([0.1, 0.1, 0.1]), diffuse=np.array([1.0, 1.0, 1.0]),
                 specular=np.array([1.0, 1.0, 1.0]), intensity: float = 1, kq=0.0019, kc=1.0, kl=0.002):
        self.specular = np.array(specular)
        self.diffuse = np.array(diffuse)
        self.ambient = np.array(ambient)
        self.pos = pos
        self.intensity = intensity
        self.kq = kq
        self.kc = kc
        self.kl = kl

    def get_illumination(self, mat, camera, point, normal):
        # Ambient
        ambient = np.multiply(mat.ambient, self.ambient)

        # Diffuse
        light_dir = normalize(self.pos - point)
        diffuse = np.multiply(
            np.multiply(mat.diffuse, self.diffuse),
            max(np.dot(normal, light_dir), 0)
        )

        # Specular (Blinn-Phong)
        view_dir = normalize(camera.pos - point)
        H = normalize(light_dir + view_dir)  # Halfway vector
        specular = np.multiply(
            np.multiply(mat.specular, self.specular),
            max(np.dot(normal, H), 0) ** mat.shininess
        )

        d = np.linalg.norm(self.pos - point)
        attenuation = 1.0 / (self.kc + self.kl * d + self.kq * d ** 2)
        return np.clip((ambient + diffuse + specular) * attenuation * self.intensity, 0, 1)


class DirectionalLight:

    def __init__(self, dir, ambient, diffuse, specular=[1, 1, 1], intensity: float = 1):
        self.specular = np.array(specular)
        self.diffuse = np.array(diffuse)
        self.ambient = np.array(ambient)
        self.dir = dir
        self.intensity = intensity

    def get_illumination(self, mat, camera, point, normal):
        # Ambient
        ambient = np.multiply(mat.ambient, self.ambient)

        # Diffuse
        light_dir = normalize(self.pos - point)
        diffuse = np.multiply(
            np.multiply(mat.diffuse, self.diffuse),
            max(np.dot(normal, light_dir), 0)
        )

        # Specular (Blinn-Phong)
        view_dir = normalize(camera.pos - point)
        H = normalize(light_dir + view_dir)  # Halfway vector
        specular = np.multiply(
            np.multiply(mat.specular, self.specular),
            max(np.dot(normal, H), 0) ** mat.shininess
        )
        return ambient + diffuse + specular


class ConeLight:

    def __init__(self, pos, dir, radius, ambient, diffuse, specular=[1, 1, 1], intensity: float = 1):
        self.specular = np.array(specular)
        self.diffuse = np.array(diffuse)
        self.ambient = np.array(ambient)
        self.pos = pos
        self.dir = dir
        self.radius = radius
        self.intensity = intensity