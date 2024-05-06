import numpy as np


def Metric(x):
  R_val = 1.0
  sigma_val = 5
  v_val = 0.9

  x_s = v_val * x[0]
  r = np.sqrt((x[1]-x_s)**2 + x[2]**2 + x[3]**2)
  f = 0.5 *( np.tanh(sigma_val*(r+R_val)) - np.tanh(sigma_val*(r-R_val)))/(np.tanh(sigma_val*R_val))

  gtt = v_val **2 * f**2 - 1.0
  gxt = -v_val*f

  return np.array([
    [gtt, gxt, 0,0],
    [gxt, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
                  ])


def Hamiltonian(x, p):
  g_inv = np.linalg.inv(Metric(x))
  return 0.5*np.dot(g_inv*p, p)


def gradient(x,p):
  eps = 0.001

  grad = (np.array([
      Hamiltonian(x,p) + np.array([eps,0,0,0]),
      Hamiltonian(x,p) + np.array([0,eps,0,0]),
      Hamiltonian(x,p) + np.array([0,0,eps,0]),
      Hamiltonian(x,p) + np.array([0,0,0,eps]),
    ]) - Hamiltonian(x,p))/eps

  return grad

def IntegrationStep(x,p):
  TimeStep = 0.1
  p = p - TimeStep * gradient(x,p)
  x = x + TimeStep * np.linalg.inv(Metric(x))*p

def getNullMomemtum(dir):
  return Metric(dir)
