import numpy as np
from color import get_pixel
from color import Image

class Camera():

  def __init__(self, position, x_pixels, y_pixels):
    self.cp = position
    self.x_pixels = x_pixels
    self.y_pixels = y_pixels
    self.y_ratio = y_pixels/x_pixels
    self.x = np.array([-1 + (2/(self.x_pixels -1))*i for i in range(self.x_pixels)])
    self.y = np.array([-1*self.y_ratio + (2/(self.y_pixels-1))*i for i in range(self.y_pixels)])

    self.ball = np.array([0,0,0])

  def rays(self):
    def soultion(vector):
      x = vector[0]
      y = vector[1]
      z = vector[2]

      discrimnant = 4*z**2 - 3
      if(discrimnant >= 0):
        return get_pixel(1,0,0)
      else:
        return get_pixel(0,0,0)


    def normalize(vector):
      magnitude = np.sqrt(sum([i**2 for i in vector]))
      return np.array([i/magnitude for i in vector])

    pixel_array = []
    for y in self.y:
      for  in self.y:
        poisition_vector = normalize(np.array([x,y,0.2]))
        pixel_array.append(soultion(poisition_vector))

    return pixel_array





if __name__ == "__main__":
  camera = Camera(np.array([0,0,1]), 320,200)
  rays = camera.rays()
  img = Image(320,200,rays)
  with open("circle.ppm", "w") as img_file:
    img.draw_image(img_file)
