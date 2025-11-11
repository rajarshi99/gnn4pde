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

import matplotlib.pyplot as plt

from golden_section_search import golden_section_search

from core.poisson_2d import Poisson_2d

import sys

def u(x,y):
    # return jnp.where(y == 1, 1, 0)
    # return jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return x**2*y**2
    # return x*y
    return 0.3*(1 - x*x - y*y)

def f_guess(x,y):
    # return 0
    # return 2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return -2*(x**2 + y**2)
    # return 0
    return 1

try:
    num_points = int(sys.argv[1])
except:
    num_points = 5

output_dir = "output/"                           # folder to save the plots 

x_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
y_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
x,y = jnp.meshgrid(x_1d,y_1d)
x = x.flatten()
y = y.flatten()

## added stuff
xy_coords = jnp.column_stack((x,y))
domain = {'vertices' : xy_coords}
for key, val in tr.triangulate(domain).items():
    domain[key] = val

p_2d = Poisson_2d(domain, u, f_guess)
K_mat, f_force, f_bound = p_2d.get_K_f1_f2()

dof_inds = jnp.array(p_2d.vert_unknown_list)
xy_int = jnp.column_stack((x[dof_inds],y[dof_inds]))
int_data_ind = jnp.where( (xy_int == jnp.array([0.5,0.5])).all(axis=1) )
u_int_data = u(xy_int[int_data_ind,0], xy_int[int_data_ind,1])

def loss(f_val):
    u_tmp = jnp.linalg.solve(K_mat, f_val*f_force - f_bound)
    deviation = u_tmp[int_data_ind] - u_int_data
    return jnp.sum(deviation*deviation)

f_optim = golden_section_search(loss, 0.8, 1.8)

f_val_list = jnp.linspace(0.8,1.8)
loss_val_list = []
for f_val in f_val_list:
    loss_val_list.append(loss(f_val))
loss_val_list = jnp.array(loss_val_list)

plt.plot(f_val_list,loss_val_list)
plt.xlabel("f")
plt.ylabel("loss")

plt.axvline(x=f_optim, linestyle="--")
plt.grid()
plt.savefig(f"{output_dir}/{num_points}_fem_inv_loss_f.png")
plt.close()

print(num_points, f_optim)
