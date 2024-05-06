import numpy as np

# cleared

class Sphere():

  def __init__(self, center, radius, material):
    self.center = center
    self.radius = radius
    self.material = material

  def intersects(self, ray):

    intersects_limit = 0.1
    current_ray = ray[:, 1:4]
    distance = np.min(np.linalg.norm(current_ray-self.center, axis=1) - intersects_limit)

    print(distance)

    if distance <= 0:
      return distance

    return None


  def normal(self, surface_point):
    surface_point_normal = surface_point - self.center
    return surface_point_normal / np.linalg.norm(surface_point_normal)
