"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufacture soution using Fourier Neural Operator
where the input are the coordinates of the nodes

Here we define
- u (unknown function)
- f (forcing function)
- x and y the coordinates of the points
"""

import jax
import jax.numpy as jnp

from core.fno_2d import FNO2d, FNO2dModel
from core.poisson_2d import Poisson_2d

import triangle as tr
import sys

import matplotlib.pyplot as plt

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

def sdf(x,y):
    dx = jnp.abs(x) - 1
    dy = jnp.abs(y) - 1

    return jnp.minimum(jnp.maximum(dx,dy),0) + jnp.sqrt(jnp.maximum(dx,0)**2 + jnp.maximum(dy,0)**2)

def main(
            num_points,
            u,
            f,
            modes1=5,
            modes2=5,
            width=5,
            activation= jnp.tanh,
            n_blocks=2,
            num_iters = 100,
            learning_rate = 5e-2,
            output_dir = "trial/"
            ):
    key_for_generating_random_numbers = jax.random.PRNGKey(69)

    plt.rcParams['font.size'] = 18
    plt.rcParams['savefig.bbox'] = 'tight'

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
    p_2d.plot_on_mesh(u_exct,
                      "",
                      f"{output_dir}{num_points}_u_exct.pgf")

    # Gloabal stiffness matrix and consistent load vector
    K_mat, f_vec = p_2d.get_K_f()

    # EXPLAIN
    Kf = jnp.hstack((K_mat,f_vec.reshape(-1,1)))

    # Input are the coordinates the shape is num_unknowns X 2
    x_int = x_1d[1:-1]
    y_int = y_1d[1:-1]
    x_int_mesh, y_int_mesh = jnp.meshgrid(x_int,y_int)
    sdf_int_mesh = sdf(x_int_mesh, y_int_mesh)
    xy_sdf_int_mesh = jnp.stack([x_int_mesh, y_int_mesh, sdf_int_mesh])

    model_key, key_for_generating_random_numbers = \
            jax.random.split(key_for_generating_random_numbers)
    fno = FNO2d(
            in_channels=3,
            out_channels=1,
            modes1=modes1,
            modes2=modes2,
            width=width,
            activation=activation,
            n_blocks=n_blocks,
            key=model_key,
            )

    # Loss is typically a func of output and target
    def loss_fn(u, Kf):
        u = u.reshape(-1,1) # Change later
        print(u.shape, Kf.shape)
        res = Kf[:,:-1] @ u - Kf[:,-1:]
        return jnp.sum(res*res)

    # Training of the FNO
    model = FNO2dModel(fno, loss_fn)
    fno, history = model.fit(xy_sdf_int_mesh, Kf,
                             learning_rate=learning_rate,
                             num_iters=num_iters)

    return fno(xy_sdf_int_mesh)

"""
    jnp.savez(f"{output_dir}details.npz",
              vertices = domain['vertices'],
              triangles = domain['triangles'],
              vertex_markers = domain['vertex_markers'],
              u_fno = u_fno,
              u_exct = u_exct,
              u_fem = u_fem
              )

    relative_l2_error = \
            jnp.linalg.norm(u_fno - u_exct) / jnp.linalg.norm(u_exct)

    return relative_l2_error
"""

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 4

    out = main(num_points, u, f, num_iters=100, learning_rate=5e-4, output_dir = "trial/")
