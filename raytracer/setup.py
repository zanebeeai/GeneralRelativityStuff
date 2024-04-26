import numpy as np
from color import Color
from point import Point
from sphere import Sphere
from ray import Ray
from scene import Scene
from render_engine import RenderEngine
from material import Material
from light import Light


WIDTH=1000
HEIGHT=1000
CAMERA=np.array([0,0,-1])
OBJECTS=[Sphere(Point(0.5,0.5,0),0.5, Material(Color.from_hex("#FF0000"))),Sphere(Point(0,0,0),0.2, Material(Color.from_hex("#FFFF00")))]
LIGHT=[Light(position=np.array([2, 2, -1]), color=Color.from_hex("#FFFFFF")), Light(position=np.array([-2, 2, -1]), color=Color.from_hex("#FFFFFF"))]
