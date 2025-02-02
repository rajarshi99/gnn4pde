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

key_for_generating_random_numbers = jax.random.PRNGKey(42)

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
    num_points = 16

output_dir = "trial/"                           # folder to save the plots 

x_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
y_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
x,y = jnp.meshgrid(x_1d,y_1d)
x = x.flatten()
y = y.flatten()

domain = {'vertices' : jnp.column_stack((x,y))}
for key, val in tr.triangulate(domain).items():
    domain[key] = val

p_2d = Poisson_2d(domain, u, f)

u_exct = u(x,y)
p_2d.plot_on_mesh(u_exct,
                  f"Exact solution {num_points}",
                  f"{output_dir}{num_points}_u_exct.png")

# Gloabal stiffness matrix and consistent load vector
K_mat, f_vec = p_2d.get_K_f()
Kf = jnp.hstack((K_mat,f_vec.reshape(-1,1)))

# Adjacency matrix using the stiffness matrix
A = jnp.where(jnp.abs(K_mat) > 1e-8, 1, 0)
deg = A.sum(axis=0)

# Guess solution shape is num_unknowns X 1
n_unknown = p_2d.n_unknown
# u_gcn = p_2d.u_known.mean() * jnp.ones(n_unknown).reshape(-1,1)
init_key, key_for_generating_random_numbers = jax.random.split(key_for_generating_random_numbers)
u_gcn = jax.random.normal(init_key, (n_unknown,1))

# GCN with 2 hidden layers,
# Last activation func as identity
# Other activations are tanh
model_key, key_for_generating_random_numbers = \
        jax.random.split(key_for_generating_random_numbers)
gcn = GCN([1, 10, 10, 1],
          [jnp.tanh] * 2 + [lambda x: x],
          model_key)

# Loss is typically a func of output and target
def loss_fn(u, Kf):
    res = Kf[:,:-1] @ u - Kf[:,-1:]
    return jnp.sum(res*res)

# Training of the GCN model
for i in range(10):
    model = GCNModel(gcn, loss_fn)
    gcn = model.fit(u_gcn, A, deg, Kf, learning_rate=5e-2, num_iters=100)
    u_gcn = gcn(u_gcn, A, deg)
u_gcn = p_2d.assemble_sol(u_gcn.reshape(-1))

p_2d.plot_on_mesh(u_gcn,
                  f"GCN solution {num_points}",
                  f"{output_dir}{num_points}_u_gcn.png")

u_fem = p_2d.get_u_fem()
p_2d.plot_on_mesh(u_fem,
                  f"FEM solution {num_points}",
                  f"{output_dir}{num_points}_u_fem.png")

p_2d.plot_on_mesh(jnp.abs(u_exct - u_fem),
                  f"|Exact solution - FEM solution| {num_points}",
                  f"{output_dir}{num_points}_u_fem_err.png")

p_2d.plot_on_mesh(jnp.abs(u_exct - u_gcn),
                  f"|Exact solution - GCN solution| {num_points}",
                  f"{output_dir}{num_points}_u_gcn_err.png")
