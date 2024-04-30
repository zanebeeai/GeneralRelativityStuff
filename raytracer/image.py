import numpy as np


class Image():

  def __init__(self, rows, columns):
    self.rows = rows
    self.columns = columns
    self.pixels = np.array([[None for _ in range(rows)] for _ in range(columns)] )

  def set_pixels(self, photo):
    self.pixels = photo

  def draw_image(self, img_file):

    img_file.write(f'P3 {self.rows} {self.columns}\n255\n')

    for i in range(self.columns):
      for p in self.pixels[i]:
        pixel_display = f'{str(int(p.red*255))} {str(int(p.green*255))} {str(int(p.blue*255))} '
        img_file.write(pixel_display)
      img_file.write('\n')
