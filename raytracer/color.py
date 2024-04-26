import numpy as np
import random as rd
from image import Image


class Pixel():

  def __init__(self,R,G,B):
    self.red = R
    self.green = G
    self.blue = B

    self.pixel = (R,G,B)

  def __repr__(self):
    return str(self.pixel)

  def __add__(self, other):
    return Pixel(self.red + other.red, self.green + other.green, self.blue + other.blue)

  def __mul__(self, other):
    return Pixel(self.red*other, self.green*other, self.blue*other)
  def __rmul__(self, other):
    return Pixel(self.red*other, self.green*other, self.blue*other)


class Color:

  def get_pixel(r,g,b):

    red = Pixel(1,0,0)
    green = Pixel(0,1,0)
    blue = Pixel(0,0,1)

    return r*red + g*green + b*blue

  @classmethod
  def from_hex(cls, hexcolor="#000000"):
    x = int(hexcolor[1:3], 16)/255.0
    y = int(hexcolor[3:5],16)/255.0
    z = int(hexcolor[5:7],16)/255.0

    return Color.get_pixel(x,y,z)

if __name__ == "__main__":

  rows = 1000
  columns = 1000
  p =[Color.get_pixel(0, 0, 0) for i in range(rows*columns)]
  test = Image(rows,columns,p)
  # print(test.pixels)
  with open("test.ppm", "w") as img_file:
    test.draw_image(img_file)
