import numpy as np
from color import Color
from point import Point
from sphere import Sphere
from ray import Ray
from scene import Scene
from render_engine import RenderEngine
from material import Material
from light import Light


# GR setup
ITERATIONS=1000
VS=0.9
SIGMA=5
R=1
C=1
DL = 0.001

# NORMAL SETUP
WIDTH=15
HEIGHT=20
CAMERA=np.array([0,0,-1])
OBJECTS=[Sphere(Point(0,0,1),0.25, Material(Color.from_hex("#FF0000")))]
LIGHT=[Light(position=np.array([2, 2, -1]), color=Color.from_hex("#FFFFFF")), Light(position=np.array([-2, 2, -1]), color=Color.from_hex("#FFFFFF"))]
