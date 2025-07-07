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

B1 = A[vert_subdom1[:,None], vert_subdom1[None,:]]
B2 = A[vert_subdom2[:,None], vert_subdom2[None,:]]

E1 = A[vert_subdom1[:,None], vert_interface[None,:]]
E2 = A[vert_subdom2[:,None], vert_interface[None,:]]

F1 = A[vert_interface[:,None], vert_subdom1[None,:]]
F2 = A[vert_interface[:,None], vert_subdom2[None,:]]

f1 = b[vert_subdom1]
f1 = b[vert_subdom2]
