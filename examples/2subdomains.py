"""
Example file for core/poisson_2d.py

Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufacture soution using FEM
Here we define
- u (unknown function)
- f (forcing function)
- x and y the coordinates of the points
"""
import jax.numpy as jnp
import triangle as tr

from core.poisson_2d import Poisson_2d
from jax.scipy.linalg import solve


import sys

def u(x,y):
    # return jnp.where(y == 1, 1, 0)
    # return jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return x**2*y**2
    # return x*y
    return 0.25*(1 - x*x - y*y)

def f(x,y):
    # return 0
    # return 2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return -2*(x**2 + y**2)
    # return 0
    return 1

num_points = 10

x_1d = jnp.linspace(-1,1,2*num_points+1, dtype=jnp.float32)
y_1d = jnp.linspace(0,1,num_points, dtype=jnp.float32)
x,y = jnp.meshgrid(x_1d,y_1d)
x = x.flatten()
y = y.flatten()

## added stuff
domain = {'vertices' : jnp.column_stack((x,y))}
for key, val in tr.triangulate(domain).items():
    domain[key] = val
    print(key, val.shape)

p_2d = Poisson_2d(domain, u, f)
A, b = p_2d.get_K_f()
vert_list = jnp.array(p_2d.vert_unknown_list)

vert_x_coords = domain["vertices"][vert_list, 0]

vert_interface = jnp.where(vert_x_coords == 0)[0]
vert_subdom1 = jnp.where(vert_x_coords < 0)[0]
vert_subdom2 = jnp.where(vert_x_coords > 0)[0]

# print("vert x coords: \n", vert_x_coords)
# print("shape: \n", vert_x_coords.shape)
# print("vert_interface = ", vert_interface)
# print("vert_subdom1: ", vert_subdom1)
# print("vert_subdom2: ", vert_subdom2)

B1 = A[vert_subdom1[:,None], vert_subdom1[None,:]]
B2 = A[vert_subdom2[:,None], vert_subdom2[None,:]]

E1 = A[vert_subdom1[:,None], vert_interface[None,:]]
E2 = A[vert_subdom2[:,None], vert_interface[None,:]]

F1 = A[vert_interface[:,None], vert_subdom1[None,:]]
F2 = A[vert_interface[:,None], vert_subdom2[None,:]]

f1 = b[vert_subdom1]
f2 = b[vert_subdom2]

C = A[vert_interface[:,None], vert_interface[None,:]]

G = b[vert_interface]

E1_ = solve(B1, E1)
E2_ = solve(B2, E2)

F1_E1_ = F1 @ E1_
F2_E2_ = F2 @ E2_

S = C - (F1_E1_ + F2_E2_)

B1_inv_f1 = solve(B1, f1)
B2_inv_f2 = solve(B2, f2)

G_ = G - (F1 @ B1_inv_f1 + F2 @ B2_inv_f2)

y = solve(S, G_)

x1 = solve(B1, f1 - E1 @ y)
x2 = solve(B2, f2 - E2 @ y)

coords = domain["vertices"]
x_all, y_all = coords[:,0], coords[:,1]
u_exact = u(x_all, y_all)

u_sol = u_exact # Somehow x_all, y_all, u_exact are numpy.ndarray
u_sol[vert_subdom1] = x1
u_sol[vert_subdom2] = x2
u_sol[vert_interface] = y

error = jnp.abs(u_sol - u_exact)
l2_error = jnp.sqrt(jnp.sum(error**2) / len(error))

print("L2 error:", l2_error)
