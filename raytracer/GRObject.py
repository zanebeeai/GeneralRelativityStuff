
# data structure for general relativity objects
class GRObject():

  def __init__(self, dptdl, dpxdl, dpydl, dpzdl, pt1, pt2, vs, R, sigma, c, iterations, dL):

      self.dptdl = dptdl
      self.dpxdl = dpxdl
      self.dpydl = dpydl
      self.dpzdl = dpzdl
      self.pt1 = pt1
      self.pt2 = pt2
      self.vs_val = vs
      self.R_val = R
      self.sigma_val = sigma
      self.c_val = c
      self.iterations = iterations
      self.dL = dL
