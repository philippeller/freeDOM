import numpy as np
from icosphere import icosphere

def get_spherical_detector(radius=20, subdivisions=12):
    vertices, faces = icosphere(subdivisions)
    return vertices * radius


def get_box_detector(x=np.linspace(-50, 50, 5), y=np.linspace(-50, 50, 5), z=np.linspace(-100, 100, 30)):
    xx, yy, zz = np.meshgrid(x, y, z)
    detector = np.zeros((xx.size, 3))
    detector[:,0] = xx.flatten()
    detector[:,1] = yy.flatten()
    detector[:,2] = zz.flatten()
    return detector