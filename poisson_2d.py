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

try:
    num_points = int(sys.argv[1])
except:
    num_points = 32

output_dir = "trial/"                           # folder to save the plots 

x_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
y_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
x,y = jnp.meshgrid(x_1d,y_1d)
x = x.flatten()
y = y.flatten()

## added stuff
domain = {'vertices' : jnp.column_stack((x,y))}
for key, val in tr.triangulate(domain).items():
    domain[key] = val
    print(key)

p_2d = Poisson_2d(domain, u, f)

u_exct = u(x,y)
p_2d.plot_on_mesh(u_exct, f"Exact solution {num_points}", f"{output_dir}{num_points}_u_exct.png")
u_sol = p_2d.sol_FEM()
p_2d.plot_on_mesh(u_sol, f"FEM solution {num_points}", f"{output_dir}{num_points}_u_fem.png")
p_2d.plot_on_mesh(u_exct - u_sol, f"Exact solution - FEM solution {num_points}", f"{output_dir}{num_points}_u_err.png")

print(p_2d.time_logs)

res = jnp.dot(p_2d.K_glob,p_2d.u_unknown) - p_2d.f_glob
print(jnp.sum(res**2))
