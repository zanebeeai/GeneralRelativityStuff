

class Material():

  def __init__(self, color, specular=0.7, diffuse=0.85, ambient=0.05, reflection=0.5):
    self.color = color
    self.specular = specular
    self.ambient = ambient
    self.diffuse = diffuse
    self.reflection = reflection 
