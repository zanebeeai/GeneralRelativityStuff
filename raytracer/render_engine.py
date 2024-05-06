import time
import numpy as np
from ray import Ray
from color import Color
from image import Image
import plotly.graph_objects as go

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
    gr_object = scene.GR

    # replace this with setup for geodesic

    geodesicArray = np.zeros((width * height, 8))
    geodesicArray[:, 0] = 0
    geodesicArray[:, 1:4] = camera # camera position

    positionArray = np.zeros((width * height, 3))

    positionArray[:, 0] = np.arange(0, width*height, 1) % width
    positionArray[:, 0] = x0 + positionArray[:, 0] * xstep

    positionArray[:, 1] = np.arange(0, width*height, 1) // width
    positionArray[:, 1] = y0 + positionArray[:, 1] * ystep

    positionArray[:, :] = positionArray[:,:] - camera

    geodesicArray[:, 5:] = positionArray/np.linalg.norm(positionArray, axis=1, keepdims=True)

    args = (
            geodesicArray[:,0],geodesicArray[:, 1], geodesicArray[:, 2], geodesicArray[:, 3],
            geodesicArray[:, 5], geodesicArray[:, 6],
            geodesicArray[:, 7], gr_object.vs_val, gr_object.R_val, gr_object.sigma_val, gr_object.c_val
            )
    geodesicArray[:, 4] = gr_object.pt1(*args)  # initial null momentum

    print("starting geodesic computation")

    geodesics = self.evaluate_geodesic(geodesicArray, scene)

    geodesics = geodesics.transpose(1, 0, 2) # tranpose to get for pixel it's geodesic path -> (H*W, iterations, 8)

    pixel_array = self.ray_trace(geodesics, scene)

    # reshape pixel_array into height x width array
    img.pixels = np.array(pixel_array).reshape((height, width)) # set image pixels




    return img

  def evaluate_geodesic(self, geodesicArray, scene):
    gr_object = scene.GR

    dptdl = gr_object.dptdl
    dpxdl = gr_object.dpxdl
    dpydl = gr_object.dpydl
    dpzdl = gr_object.dpzdl


    vs_val = gr_object.vs_val
    R_val = gr_object.R_val
    sigma_val = gr_object.sigma_val
    c_val = gr_object.c_val

    iterations = gr_object.iterations
    dL = gr_object.dL

    geodesics = [ ]
    print("starting geodesic computation ... \n")
    start = time.time()

    for i in range(iterations):

      args =  (
              geodesicArray[:,0],geodesicArray[:, 1], geodesicArray[:, 2], geodesicArray[:, 3],
              geodesicArray[:, 4], geodesicArray[:, 5], geodesicArray[:, 6],
              geodesicArray[:, 7], vs_val, R_val, sigma_val, c_val
              )

      dp = np.array([dptdl(*args), dpxdl(*args), dpydl(*args), dpzdl(*args)]).T
      geodesicArray[:, 4:] += dL * dp
      geodesicArray[:, :4] += geodesicArray[:, 4:] * dL

      geodesics.append(np.copy(geodesicArray))
      print("Percentage done: {:3.5f} %".format((i/iterations)*100), end="\r")

    print("Elapsed time: {:3.2f} seconds".format(time.time()-start))
    geodesics = np.array(geodesics)
    print(geodesics.shape)
    self.plot(geodesics, scene)

    return geodesics

  def plot(self, geods, scene):

    geods_transposed = geods.transpose(1, 0, 2)
    fig = go.Figure()

    for geod_angle in geods_transposed:
      positions = geod_angle[:, 0:4]
      velocities = geod_angle[:, 4:]

      ts, xs, ys, zs = zip(*positions)
      tps, xps, yps, zps =  zip(*velocities)

      # Plot for position
      fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines'))

      # Add initial velocity vector
      fig.add_trace(go.Cone(x=[xs[0]], y=[ys[0]], z=[zs[0]],
                  u=[xps[0]], v=[yps[0]], w=[zps[0]],
                  sizemode="scaled",
                  sizeref=0.01,
                  anchor="tail"))

    print("plotting ...")

    def generate_sphere(sphere, color):
      phi = np.linspace(0, 2 * np.pi, 100)
      theta = np.linspace(0, np.pi, 100)
      phi, theta = np.meshgrid(phi, theta)
      x_sphere= sphere.center[0] + sphere.radius * np.sin(theta) * np.cos(phi)
      y_sphere = sphere.center[1] + sphere.radius * np.sin(theta) * np.sin(phi)
      z_sphere = sphere.center[2] + sphere.radius * np.cos(theta)

      return go.Mesh3d(
          x=x_sphere.flatten(), y=y_sphere.flatten(), z=z_sphere.flatten(),
          alphahull=0,
          opacity=0.3,
          color=color
      )

# Inner sphere (radius 0.5)
    fig.add_trace(generate_sphere(scene.objects[0], 'lime'))


    fig.update_layout(scene = dict(
              xaxis_title='X Position (x)',
              yaxis_title="Y Position (y)",
              zaxis_title="z Position (z)"),
              width=700,
              margin=dict(r=20, b=10, l=10, t=10))

    fig.show()

  def ray_trace(self, geods, scene):

    pixels = np.zeros((scene.width * scene.height), dtype=object)

    for i in range(scene.width * scene.height):

      pixels[i] = self.find_nearest(geods[i], scene)
      print("Percentage done for intersection: {:3.5f} %".format((i/(scene.width * scene.height))*100), end="\r")

    return pixels

  def find_nearest(self, ray, scene):
    dist_min = None
    obj_hit = None

    for obj in scene.objects:
      dist = obj.intersects(ray)
      if dist and (dist_min is None or dist < dist_min):
        dist_min = dist
        obj_hit = obj
        

    return Color.get_pixel(0,0,0) if obj_hit is None else Color.get_pixel(0,0,0) +  self.color_at(obj_hit)

  def color_at(self, obj_hit):

    material = obj_hit.material
    obj_color = material.color
    # to_cam = scene.camera - hit_pos
    # color = material.ambient * Color.from_hex("#000000")
    # for light in scene.light:
    #   light_ray = Ray(hit_pos, np.array(light.position) - hit_pos)
    #   color += obj_color * material.diffuse * max(np.dot(hit_pos_normal, light_ray.direction), 0)
    #   half_vector = (light_ray.direction + to_cam)/np.linalg.norm(light_ray.direction + to_cam)
    #   color += obj_color * material.specular * max(np.dot(hit_pos_normal, half_vector), 0) ** 50

    return obj_color
