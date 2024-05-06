
# local imports

from color import Color
from point import Point
from sphere import Sphere
from ray import Ray
from scene import Scene
from render_engine import RenderEngine
from material import Material
from light import Light
from GRObject import GRObject
import setup


# mathamatical imports (external)
import sympy as sp
import numpy as np
from itertools import product

# cleared

def main():

  print("begining to load symbols")
  vs, sigma, R, lam, c= sp.symbols('v_s sigma R lambda c')

  t = sp.Function('t')(lam)
  x = sp.Function('x')(lam)
  y = sp.Function('y')(lam)
  z = sp.Function('z')(lam)

  dt=sp.diff(t)
  dx=sp.diff(x)
  dy=sp.diff(y)
  dz=sp.diff(z)

  pt = sp.diff(t, lam)
  px = sp.diff(x, lam)
  py = sp.diff(y, lam)
  pz = sp.diff(y, lam)

  dptdl = sp.sympify(open("dptdl.txt", "r").read())
  dpxdl = sp.sympify(open("dpxdl.txt", "r").read())
  dpydl = sp.sympify(open("dpydl.txt", "r").read())
  dpzdl = sp.sympify(open("dpzdl.txt", "r").read())
  pt1 = sp.sympify(open("pt1.txt", "r").read())
  pt2 = sp.sympify(open("pt2.txt", "r").read())

  dptdl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs,R,sigma,c), dptdl, "numpy")
  dpxdl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs,R,sigma,c), dpxdl, "numpy")
  dpydl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs,R,sigma,c), dpydl, "numpy")
  dpzdl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs,R,sigma,c), dpzdl, "numpy")


  pt1_lambda = sp.lambdify((t,x,y,z,vs,R,sigma,c,px,py,pz), pt1, "numpy")
  pt2_lambda = sp.lambdify((t,x,y,z,vs,R,sigma,c,px,py,pz), pt2, "numpy")

  vs_val = setup.VS
  R_val = setup.R
  sigma_val = setup.SIGMA
  c_val = setup.C
  iterations = setup.ITERATIONS
  dL = setup.DL

  print("creating GR object")

  GRObject1 = GRObject(dptdl_lambda, dpxdl_lambda, dpydl_lambda, dpzdl_lambda, pt1_lambda, pt2_lambda, vs_val, R_val, sigma_val, c_val, iterations, dL)


  width = setup.WIDTH
  height = setup.HEIGHT

  camera = setup.CAMERA
  objects = setup.OBJECTS
  light = setup.LIGHT

  scene = Scene(camera, objects, width, height, light, GRObject1)

  engine = RenderEngine()
  image = engine.render_scene(scene)

  with open("test.ppm", "w") as img_file:
    image.draw_image(img_file)

if __name__ == "__main__":
  main()
