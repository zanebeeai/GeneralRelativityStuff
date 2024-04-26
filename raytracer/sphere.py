import numpy as np

# cleared

class Sphere():

  def __init__(self, center, radius, material):
    self.center = center
    self.radius = radius
    self.material = material

  def intersects(self, ray):

    oc = ray.origin - self.center
    a = 1.0
    b = 2.0 * np.dot(oc, ray.direction)
    c = np.dot(oc, oc) - self.radius**2
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
      temp = (-b - np.sqrt(discriminant))/(2*a)
      if temp > 0:
        return temp
    return None


  def normal(self, surface_point):
    surface_point_normal = surface_point - self.center
    return surface_point_normal / np.linalg.norm(surface_point_normal)

  
