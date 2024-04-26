import numpy as np
from color import Color
from point import Point
from sphere import Sphere
from ray import Ray
from scene import Scene
from render_engine import RenderEngine
from material import Material
from light import Light
import setup

# cleared

def main():
  width = setup.WIDTH
  height = setup.HEIGHT

  camera = setup.CAMERA
  objects = setup.OBJECTS
  light = setup.LIGHT
  scene = Scene(camera, objects, width, height, light)
  engine = RenderEngine()
  image = engine.render_scene(scene)

  with open("test.ppm", "w") as img_file:
    image.draw_image(img_file)

if __name__ == "__main__":
  main()
