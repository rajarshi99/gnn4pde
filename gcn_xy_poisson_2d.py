"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufacture soution using message passing by GCN
where the input are the coordinates of the nodes

Here we define
- u (unknown function)
- f (forcing function)
- x and y the coordinates of the points
"""
import jax
import jax.numpy as jnp

from core.gcn import GCN, GCNModel
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

def main(
            num_points,
            u,
            f,
            gcn_layers = [2, 10, 10, 1],
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

    # Adjacency matrix for interior points
    A = jnp.zeros_like(K_mat, dtype=jnp.int8)
    A = A.at[jnp.nonzero(K_mat)].set(1)
    d = A.sum(axis=0)

    # Input are the coordinates the shape is num_unknowns X 2
    vert_unknown_list = jnp.array(p_2d.vert_unknown_list)
    x = x.at[vert_unknown_list].get()
    y = y.at[vert_unknown_list].get()
    xy = jnp.column_stack((x, y))

    # GCN with 2 hidden layers,
    # Last activation func as identity
    # Other activations are tanh
    model_key, key_for_generating_random_numbers = \
            jax.random.split(key_for_generating_random_numbers)
    gcn = GCN(gcn_layers,
              # [jnp.tanh] * (len(gcn_layers)-2) + [lambda x: x],
              [jax.nn.relu] * (len(gcn_layers)-1),
              model_key)

    # Loss is typically a func of output and target
    def loss_fn(u, Kf):
        res = Kf[:,:-1] @ u - Kf[:,-1:]
        return jnp.sum(res*res)

    # Training of the GCN model
    model = GCNModel(gcn, loss_fn)
    gcn, history = model.fit(xy, A, d, Kf,
                             learning_rate=learning_rate,
                             num_iters=num_iters)

    u_gcn = gcn(xy, A, d).flatten()
    u_gcn = p_2d.assemble_sol(u_gcn)
    p_2d.plot_on_mesh(u_gcn,
                      f"GCN solution",
                      f"{output_dir}{num_points}_u_gcn.png")
    p_2d.plot_on_mesh(u_gcn,
                      "",
                      f"{output_dir}{num_points}_u_gcn.pgf")

    p_2d.plot_on_mesh(u_gcn,
                      f"GCN solution",
                      f"{output_dir}{num_points}_u_gcn_with_mesh.png",
                      plot_with_lines = True)
    p_2d.plot_on_mesh(u_gcn,
                      "",
                      f"{output_dir}{num_points}_u_gcn_with_mesh.pgf",
                      plot_with_lines = True)

    u_fem = p_2d.get_u_fem()
    p_2d.plot_on_mesh(u_fem,
                      f"FEM solution",
                      f"{output_dir}{num_points}_u_fem.png")
    p_2d.plot_on_mesh(u_fem,
                      "",
                      f"{output_dir}{num_points}_u_fem.pgf")

    p_2d.plot_on_mesh(jnp.abs(u_exct - u_fem),
                      f"|Exact solution - FEM solution|",
                      f"{output_dir}{num_points}_u_fem_err.png")
    p_2d.plot_on_mesh(jnp.abs(u_exct - u_fem),
                      "",
                      f"{output_dir}{num_points}_u_fem_err.pgf")

    p_2d.plot_on_mesh(jnp.abs(u_exct - u_gcn),
                      f"|Exact solution - GCN solution|",
                      f"{output_dir}{num_points}_u_gcn_err.png")
    p_2d.plot_on_mesh(jnp.abs(u_exct - u_gcn),
                      "",
                      f"{output_dir}{num_points}_u_gcn_err.pgf")

    # Plot the loss history
    iter_ids = history["iter_ids"]
    loss_vals = history["loss_vals"]
    plt.plot(iter_ids, loss_vals, color="k")

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_gcn.png")
    plt.savefig(f"{output_dir}{num_points}_loss_gcn.pgf")
    plt.close()

    jnp.savez(f"{output_dir}details.npz",
              vertices = domain['vertices'],
              triangles = domain['triangles'],
              vertex_markers = domain['vertex_markers'],
              u_gcn = u_gcn,
              u_exct = u_exct,
              u_fem = u_fem
              )

    relative_l2_error = \
            jnp.linalg.norm(u_gcn - u_exct) / jnp.linalg.norm(u_exct)

    return relative_l2_error

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 4

    main(num_points, u, f, num_iters=10000, learning_rate=5e-4, output_dir = "trial/")
