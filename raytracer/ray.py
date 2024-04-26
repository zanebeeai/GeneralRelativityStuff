import numpy as np

# cleared

class Ray():

  def __init__(self, origin, direction):

    def normalize(vector):
      magnitude = np.sqrt(np.dot(vector, vector))
      return np.array([i/magnitude for i in vector])

    self.origin = origin
    self.direction = normalize(direction)
