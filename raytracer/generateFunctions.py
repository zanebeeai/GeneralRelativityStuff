
import sympy as sp
import numpy as np
from itertools import product

def getMetric(lineElement, coordSystem="Cartesian", subs=None, overrideConst = False): #the override lets the code run faster if you know for sure your line element will work out
    if coordSystem not in ["Cartesian", "PlanePolar", "SphericalPolar", "CylindricalPolar"]:
        raise ValueError("Unknown coordinate system")

    lineElement=sp.expand(lineElement)
    coords = (t, x, y, z)

    dim = len(coords)
    g = sp.zeros(dim)

    for mu in range(dim):
        for nu in range(dim):
            coeff = lineElement.coeff(sp.diff(coords[mu]) * sp.diff(coords[nu]))
            if mu != nu and coeff != 0:
                g[mu, nu] = coeff.subs(subs) / 2
            else:
                g[mu, nu] = coeff.subs(subs)

    # Check for unexpected terms in the line element
    if not overrideConst:
        reconstructed_line_element = sum(g[i, j] * sp.diff(coords[i]) * sp.diff(coords[j]) for i in range(dim) for j in range(dim))
        if sp.simplify(lineElement.subs(subs) - reconstructed_line_element) != 0:
            raise ValueError("Line element contains terms that are not pure differentials of the coordinates used")
    return g

print("starting ... ")
vs, sigma, R, lam = sp.symbols('v_s sigma R lambda')
t = sp.Function('t')(lam)
x = sp.Function('x')(lam)
y = sp.Function('y')(lam)
z = sp.Function('z')(lam)

dt=sp.diff(t)
dx=sp.diff(x)
dy=sp.diff(y)
dz=sp.diff(z)

# for now, define constants as symbols
xs, r, f_r, c = sp.symbols("x_s r f_r c")

lineElement = -c**2*dt**2 + (dx - vs*f_r*dt)**2 + dy**2 + dz**2

# order of substitutions matter!!!
subs= [
    (f_r, (sp.tanh(sigma * (r + R)) - sp.tanh(sigma * (r - R))) / (2 * sp.tanh(sigma * R))),
    (r, sp.sqrt((x - xs)**2 + y**2 + z**2)),
    (xs, vs*t), # since its steady state
]


metric=getMetric(lineElement, "Cartesian", subs, True)

print("got metric")



metric_inv = metric.inv()
print("start computation")
n=4
X = [t, x, y, z]
# computing the symbols using the metric equation
# Create array to store the computed christoffel symbols.
christoffel_symbols = np.zeros(shape=n, dtype='object')
simple = False
for i in range(n):
    dummy_matrix = sp.Matrix.zeros(n,n)
    for (j,k,l) in product(range(n), repeat=3):
        dummy_matrix[j,k] += (
            sp.Rational(1/2)*metric_inv[i,l] * (sp.diff(metric[l,j],X[k])
            +sp.diff(metric[l,k],X[j]) - sp.diff(metric[j,k],X[l]))
        )
        print(f"done connection j: {j} k: {k} l: {l}")
    christoffel_symbols[i] = sp.simplify(dummy_matrix) if simple else dummy_matrix

C =  christoffel_symbols

""" Define momentum vector in terms of lambda """

pt = sp.diff(t, lam)
px = sp.diff(x, lam)
py = sp.diff(y, lam)
pz = sp.diff(y, lam)


""" Compute change in momentum vector with respect to lambda """

Ct, Cx, Cy, Cz = C[0], C[1], C[2], C[3]
dptdl = -1 * (Ct[0,0] * pt**2 + Ct[1,1] * px**2 + Ct[2,2] * py**2 + Ct[3,3] * pz**2 + 2 * (Ct[0,1] * pt * px + Ct[0,2] * pt * py + Ct[0,3] * pt * pz + Ct[1,2] * px * py + Ct[1,3] * px * pz + Ct[2,3] * py * pz))
dpxdl = -1 * (Cx[0,0] * pt**2 + Cx[1,1] * px**2 + Cx[2,2] * py**2 + Cx[3,3] * pz**2 + 2 * (Cx[0,1] * pt * px + Cx[0,2] * pt * py + Cx[0,3] * pt * pz + Cx[1,2] * px * py + Cx[1,3] * px * pz + Cx[2,3] * py * pz))
dpydl = -1 * (Cy[0,0] * pt**2 + Cy[1,1] * px**2 + Cy[2,2] * py**2 + Cy[3,3] * pz**2 + 2 * (Cy[0,1] * pt * px + Cy[0,2] * pt * py + Cy[0,3] * pt * pz + Cy[1,2] * px * py + Cy[1,3] * px * pz + Cy[2,3] * py * pz))
dpzdl = -1 * (Cz[0,0] * pt**2 + Cz[1,1] * px**2 + Cz[2,2] * py**2 + Cz[3,3] * pz**2 + 2 * (Cz[0,1] * pt * px + Cz[0,2] * pt * py + Cz[0,3] * pt * pz + Cz[1,2] * px * py + Cz[1,3] * px * pz + Cz[2,3] * py * pz))


Bquad=2*(metric[0, 1]*px+metric[0, 2]*py+metric[0, 3]*pz)
Cquad=metric[1, 1]*px**2+metric[2, 2]*py**2+metric[3, 3]*pz**2+2*(
      metric[1, 2]*px*py+metric[1, 3]*px*pz+metric[2, 3]*py*pz
)

pt1 = (-Bquad+sp.sqrt(Bquad**2-4*metric[0,0]*Cquad))/(2*metric[0,0])
pt2 = ((-Bquad - sp.sqrt(Bquad**2 - 4 * metric[0, 0] * Cquad)) / (2 * metric[0, 0]))



with open(str("dptdl") + ".txt", "w") as f:
  f.write(sp.srepr(dptdl))

with open(str("dpxdl") + ".txt", "w") as f:
  f.write(sp.srepr(dpxdl))

with open(str("dpydl") + ".txt", "w") as f:
  f.write(sp.srepr(dpydl))

with open(str("dpzdl") + ".txt", "w") as f:
  f.write(sp.srepr(dpzdl))

with open(str("pt1") + ".txt", "w") as f:
  f.write(sp.srepr(pt1))

with open(str("pt2") + ".txt", "w") as f:
  f.write(sp.srepr(pt2))
