import numpy as np

# cleared

def Point(x, y, z):
  return np.array([x,y,z])


if __name__ == "__main__":
  p = Point(1,2,3)
  print(p)
