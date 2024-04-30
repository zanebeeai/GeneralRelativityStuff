

class geodesicObject():

  def __init__(self, pos, symbols, d_pos, dp, iterations):


    self.vs = symbols[0]
    self.sigma = symbols[1]
    self.R = symbols[2]

    self.t = pos[0]
    self.x = pos[1]
    self.y = pos[2]
    self.z = pos[3]

    self.dt = d_pos[0]
    self.dx = d_pos[1]
    self.dy = d_pos[2]
    self.dz = d_pos[3]


    self.dptdl = dp[0]
    self.dpxdl = dp[1]
    self.dpydl = dp[2]
    self.dpzdl = dp[3]

    self.iterations = iterations
