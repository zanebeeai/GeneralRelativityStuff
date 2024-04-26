
import numpy as np
from ray import Ray
from color import Color
from image import Image

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

    for j in range(height):
      y = y0 + j*ystep
      for i in range(width):
        x = x0 + i*xstep
        ray = Ray(camera, np.array([x, y, 0]) - camera)
        img.set_pixel(i, j, self.ray_trace(ray, scene))
        print("{:3.0f}%".format(float(j) / float(height) * 100), end="\r")

    return img


  def ray_trace(self, ray, scene, count=0):

    color = Color.get_pixel(0,0,0)
    obj_hit, dist = self.find_nearest(ray, scene)
    if obj_hit is None:
      return color
    hit_pos = ray.origin + ray.direction * dist
    hit_pos_normal = obj_hit.normal(hit_pos)
    color += self.color_at(obj_hit, hit_pos,hit_pos_normal, scene)

    reflected_ray = Ray(hit_pos, ray.direction - 2 * np.dot(ray.direction, hit_pos_normal) * hit_pos_normal)
    reflected_color = self.ray_trace(reflected_ray, scene, count + 1)
    color += reflected_color * obj_hit.material.reflection
    return color

  def find_nearest(self, ray, scene):
    dist_min = None
    obj_hit = None

    for obj in scene.objects:
      dist = obj.intersects(ray)
      if dist and (dist_min is None or dist < dist_min):
        dist_min = dist
        obj_hit = obj

    return (obj_hit, dist_min)

  def color_at(self, obj_hit, hit_pos,hit_pos_normal, scene):


    material = obj_hit.material
    obj_color = material.color
    to_cam = scene.camera - hit_pos
    color = material.ambient * Color.from_hex("#000000")
    for light in scene.light:
      light_ray = Ray(hit_pos, np.array(light.position) - hit_pos)
      color += obj_color * material.diffuse * max(np.dot(hit_pos_normal, light_ray.direction), 0)
      half_vector = (light_ray.direction + to_cam)/np.linalg.norm(light_ray.direction + to_cam)
      color += obj_color * material.specular * max(np.dot(hit_pos_normal, half_vector), 0) ** 50

    return color
