from color import Color


class Light():

  def __init__(self,position, color=Color.from_hex("#FFFFFF")):
    self.on = False
    self.color = color
    self.position = position
