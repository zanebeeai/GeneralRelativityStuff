# for raytracer
from ray import Ray
from color import Color
import time
from image import Image

import numpy as np

# for geodesics
import sympy as sp


class RenderEngine:

  def render_scene(self, scene):
    width = scene.width
    height = scene.height
    aspect_ratio = float(width) / height
    x0 = -1.0
    x1 = 1.0
    xstep = (x1 - x0) / (width - 1)
    y0 = -1.0 / aspect_ratio
    y1 = 1.0 / aspect_ratio
    ystep = (y1 - y0) / (height - 1)

    camera = scene.camera
    objects = scene.objects
    img = Image(width, height)

    start = time.time()

    geodesicArray = np.zeros((width*height, 8))
    geodesicArray[:, 1:4] = camera

    start2 = time.time()

    positionArray = np.zeros((width*height,3))
    positionArray[:, 0] = np.arange(0, width*height, 1)

    positionArray[:,1] = positionArray[:,0] // width

    positionArray[:,0] = positionArray[:,0] % width
    positionArray[:,0] = positionArray[:,0] * xstep + x0

    positionArray[:,1] = positionArray[:,1] * ystep + y0

    geodesicArray[:, 4:7] = positionArray - camera
    geodesicArray[:, 4:7] = geodesicArray[:, 4:7] / np.linalg.norm(geodesicArray[:, 4:7], axis=1)[:, np.newaxis]

    print("starting geodesic tracing ...")
    pixelArray = self.evaluate_geodesic(scene.GR, geodesicArray)
    print("done geodesic tracing ...")
    img = self.setImage(scene, pixelArray)

    print("done setting image ...")

    return img

  def evaluate_geodesic(self, GR, values):

    g = GR

    sigma = g.sigma
    vs = g.vs
    R = g.R

    t = g.t
    x = g.x
    y = g.y
    z = g.z

    dt = g.dt
    dx = g.dx
    dy = g.dy
    dz = g.dz

    dptdl = g.dptdl
    dpxdl = g.dpxdl
    dpydl = g.dpydl
    dpzdl = g.dpzdl

    # define constants
    sigma_val = 1
    vs_val=3
    R_val=2


    dptdl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs, sigma, R), dptdl, "numpy")
    dpxdl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs, sigma, R), dpxdl, "numpy")
    dpydl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs, sigma, R), dpydl, "numpy")
    dpzdl_lambda = sp.lambdify((t, x, y, z, dt, dx, dy, dz, vs, sigma, R), dpzdl, "numpy")


    geods = []
    geods.append(np.copy(values))

    # set up recursion

    iterations = GR.iterations
    dL = 0.01
    L = 0
    count = 0

    for i in range(iterations):

      args = (values[:, 0], values[:, 1], values[:, 2], values[:, 3], values[:, 4], values[:, 5], values[:, 6],
            values[:, 7], vs_val, sigma_val, R_val)

      dp=[
          dptdl_lambda(*args),
          dpxdl_lambda(*args),
          dpydl_lambda(*args),
          dpzdl_lambda(*args)
      ]

      values[:,4:] += dL * np.array(dp).T # update the velocities
      values[:, 0:4] += dL * values[:, 4:] # update positions
      geods.append(np.copy(values))

      count += 1
      print("Percentage done: {:3.5f} %".format((count/iterations)*100), end="\r")

    return np.array(geods) # return the geodesics

  def setImage(self, scene, pixel_array):
    img = Image(scene.width, scene.height)
    colors = self.ray_trace(pixel_array, scene)
    img.set_pixels(colors.reshape(scene.height, scene.width))

    return img


  def ray_trace(self, geods, scene, count=0):

    color_array = self.find_nearest(geods, scene)

    return color_array

  def find_nearest(self, geods, scene):
    dist_min = None
    obj_hit = None

    color = Color.get_pixel(0,0,0)
    for obj in scene.objects:
      dist = np.where(obj.intersects(geods) > 0, self.color_at(obj), color)

    return dist

  def color_at(self, obj_hit):

    material = obj_hit.material
    obj_color = material.color

    return obj_color
