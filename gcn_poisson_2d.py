"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufacture soution using message passing by GCN
Here we define
- u (unknown function)
- f (forcing function)
- x and y the coordinates of the points
"""
import jax
import jax.numpy as jnp

from scirex.core.dl.gcn import GCN, GCNModel
from core.poisson_2d import Poisson_2d

import triangle as tr
import sys

def u(x,y):
    # return jnp.where(y == 1, 1, 0)
    # return jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return x**2*y**2
    return x*y

def f(x,y):
    # return 0
    # return 2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return -2*(x**2 + y**2)
    return 0

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

## Main implementation to be added from here


