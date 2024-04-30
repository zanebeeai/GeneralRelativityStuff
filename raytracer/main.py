import numpy as np

from scene import Scene
from render_engine import RenderEngine
import setup

import sympy as sp
from grobject import geodesicObject

# cleared

def main():

    """Main function that renders a scene based off a given general relativity setup."""

    # setup geodesic metrics

    print("starting ...")
    # t, x, y, z, vs, sigma, R = sp.symbols('t x y z v_s sigma R')
    vs, sigma, R = sp.symbols('v_s sigma R')

    lam = sp.symbols('lambda')

    t = sp.Function('t')(lam)
    x = sp.Function('x')(lam)
    y = sp.Function('y')(lam)
    z = sp.Function('z')(lam)


    xs = vs * t  # since it's steady state
    r = sp.sqrt((x - xs)**2 + y**2 + z**2)
    f_r = (sp.tanh(sigma * (r + R)) - sp.tanh(sigma * (r - R))) / (2 * sp.tanh(sigma * R))

    #metric components based on the line element
    g_tt = -(1 - vs**2 * f_r**2)
    g_tx = g_xt = vs * f_r
    g_xx = g_yy = g_zz = 1

    #g_uv
    metric = sp.Matrix([[g_tt, g_tx, 0, 0],
                        [g_xt, g_xx, 0, 0],
                        [0, 0, g_yy, 0],
                        [0, 0, 0, g_zz]])
    # The inverse metric tensor g^uv (needed for Christoffel symbols)
    metric_inv = metric.inv()

    print("done metric tensor")

    # defining a partial
    def partial_derivative(matrix, var):
        """ This function returns the matrix of partial derivatives """
        return sp.Matrix(matrix.shape[0], matrix.shape[1], lambda i,j: sp.diff(matrix[i, j], var))

    # derivatives of metric tensor
    partial_t = partial_derivative(metric, t)
    print("done partial_t")

    partial_x = partial_derivative(metric, x)
    print("done partial_x")

    partial_y = partial_derivative(metric, y)
    print("done partial_y")

    partial_z = partial_derivative(metric, z)
    print("done partial_z")

    print("starting computation of Christoffel symbols...")
    # computing the symbols using the metric equation
    christoffel_symbols = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
    for lambda_ in range(4):
        for mu in range(4):
            for nu in range(4):
                christoffel_symbols[lambda_][mu][nu] = 1/2 * (
                    metric_inv[lambda_, 0] * (partial_x[mu, nu] + partial_x[nu, mu] - partial_t[mu, nu]) +
                    metric_inv[lambda_, 1] * (partial_t[mu, nu] + partial_t[nu, mu] - partial_x[mu, nu]) +
                    metric_inv[lambda_, 2] * (partial_y[mu, nu] + partial_y[nu, mu] - partial_y[mu, nu]) +
                    metric_inv[lambda_, 3] * (partial_z[mu, nu] + partial_z[nu, mu] - partial_z[mu, nu])
                ).simplify()

    C = christoffel_symbols

    print("done computing Christoffel symbols")

    d_position = np.array([sp.diff(t, lam), sp.diff(x,lam), sp.diff(y, lam), sp.diff(z, lam)])

    Ct, Cx, Cy, Cz = C[0], C[1], C[2], C[3]
    pt, px, py, pz  = d_position

    dptdl = -1 * (Ct[0][0] * pt**2 + Ct[1][1] * px**2 + Ct[2][2] * py**2 + Ct[3][3] * pz**2 + 2 * (Ct[0][1] * pt * px + Ct[0][2] * pt * py + Ct[0][3] * pt * pz + Ct[1][2] * px * py + Ct[1][3] * px * pz + Ct[2][3] * py * pz))
    dpxdl = -1 * (Cx[0][0] * pt**2 + Cx[1][1] * px**2 + Cx[2][2] * py**2 + Cx[3][3] * pz**2 + 2 * (Cx[0][1] * pt * px + Cx[0][2] * pt * py + Cx[0][3] * pt * pz + Cx[1][2] * px * py + Cx[1][3] * px * pz + Cx[2][3] * py * pz))
    dpydl = -1 * (Cy[0][0] * pt**2 + Cy[1][1] * px**2 + Cy[2][2] * py**2 + Cy[3][3] * pz**2 + 2 * (Cy[0][1] * pt * px + Cy[0][2] * pt * py + Cy[0][3] * pt * pz + Cy[1][2] * px * py + Cy[1][3] * px * pz + Cy[2][3] * py * pz))
    dpzdl = -1 * (Cz[0][0] * pt**2 + Cz[1][1] * px**2 + Cz[2][2] * py**2 + Cz[3][3] * pz**2 + 2 * (Cz[0][1] * pt * px + Cz[0][2] * pt * py + Cz[0][3] * pt * pz + Cz[1][2] * px * py + Cz[1][3] * px * pz + Cz[2][3] * py * pz))

    dptdl = dptdl.simplify()
    dpxdl = dpxdl.simplify()
    dpydl = dpydl.simplify()
    dpzdl = dpzdl.simplify()

    dp  = np.array([dptdl, dpxdl, dpydl, dpzdl])

    pos = np.array([t, x, y, z])
    symbols = np.array([vs, sigma, R])

    GR = geodesicObject(pos, symbols, d_position, dp, setup.ITERATIONS)

    print("done GR setup")

    width = setup.WIDTH
    height = setup.HEIGHT

    camera = setup.CAMERA
    objects = setup.OBJECTS
    light = setup.LIGHT
    scene = Scene(camera, objects, width, height, light, GR)

    print("starting rendering ...")

    engine = RenderEngine()
    image = engine.render_scene(scene)

    print("drawing image ...")

    with open("test.ppm", "w") as img_file:
        image.draw_image(img_file)

if __name__ == "__main__":
  main()
