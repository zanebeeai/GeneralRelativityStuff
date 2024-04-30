import numpy as np 
# cleared

class Ray():

  def __init__(self, origin, direction):



    self.origin = origin
    self.direction = direction/np.linalg.norm(direction)
