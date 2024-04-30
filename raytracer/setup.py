from color import Color
from point import Point
from sphere import Sphere
from material import Material
from light import Light

import numpy as np

WIDTH=60
HEIGHT=40
CAMERA=np.array([0,0,-1])
OBJECTS=[Sphere(Point(0,0,0),1, Material(Color.from_hex("#FFFF00")))]
LIGHT=[Light(position=np.array([2, 2, -1]), color=Color.from_hex("#FFFFFF")), Light(position=np.array([-2, 2, -1]), color=Color.from_hex("#FFFFFF"))]
ITERATIONS=1000
