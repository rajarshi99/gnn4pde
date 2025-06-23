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
            u = u,
            f = f,
            gcn_layers = [2, 10, 10, 1],
            num_iters = 1000,
            learning_rate = 5e-2,
            output_dir = "trial/"
            ):
    key_for_generating_random_numbers = jax.random.PRNGKey(69)

    x_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
    y_1d = jnp.linspace(-1,1,num_points, dtype=jnp.float32)
    x,y = jnp.meshgrid(x_1d,y_1d)
    x = x.flatten()
    y = y.flatten()
    xy = jnp.column_stack((x, y))

    domain = {'vertices' : jnp.column_stack((x,y))}
    for key, val in tr.triangulate(domain).items():
        domain[key] = val

    p_2d = Poisson_2d(domain, u, f)

    vert_unknown_list = jnp.array(p_2d.vert_unknown_list)
    x_int = x.at[vert_unknown_list].get()
    y_int = y.at[vert_unknown_list].get()
    xy_int = jnp.column_stack((x_int, y_int))

    u_exct = u(x,y)
    p_2d.plot_on_mesh(u_exct,
                      f"Exact solution {num_points}",
                      f"{output_dir}{num_points}_u_exct.png")

    # Gloabal stiffness matrix and consistent load vector
    n_nodes = p_2d.u_sol.shape[0]
    K_mat, f_vec = p_2d.get_K_f()

    # Interior adjacency matrix using the stiffness matrix
    A_int = jnp.zeros_like(K_mat, dtype=jnp.int8)
    A_int = A_int.at[jnp.nonzero(K_mat)].set(1)
    d_int = A_int.sum(axis=0)

    # EXPLAIN
    K_glob_sparse = p_2d.K_glob_sparse
    K_full = jnp.zeros((n_nodes,n_nodes))
    K_full = K_full.at[tuple(zip(
        *K_glob_sparse["ind"]
        ))].set(K_glob_sparse["val"])

    # EXPLAIN
    f_glob_sparse = p_2d.f_glob_sparse
    f_full = jnp.zeros(n_nodes)
    f_full = f_full.at[jnp.array(
        f_glob_sparse["ind"]
        )].set(f_glob_sparse["val"])

    # EXPLAIN
    Kf = jnp.hstack((K_full,f_full.reshape(-1,1)))

    # Adjacency matrix
    A_full = jnp.zeros((n_nodes,n_nodes), dtype=jnp.int8)
    A_full = A_full.at[tuple(zip(
        *K_glob_sparse["ind"]
        ))].set(1)
    d_full = A_full.sum(axis=0)
    d_full = jnp.where(d_full == 0, 1, d_full)

    # GCN with 2 hidden layers,
    # Last activation func as identity
    # Other activations are tanh
    model_key, key_for_generating_random_numbers = \
            jax.random.split(key_for_generating_random_numbers)
    gcn = GCN(gcn_layers,
              [jnp.tanh] * (len(gcn_layers)-2) + [lambda x: x],
              model_key)

    # Loss is typically a func of output and target
    def loss_fn(u, Kf):
        res = Kf[:,:-1] @ u - Kf[:,-1:]
        return jnp.sum(res*res)

    # Training of the GCN model
    history_list = []
    model = GCNModel(gcn, loss_fn) # Change operation to go before for
    gcn, history = model.fit(xy, A_full, d_full, Kf,
                             learning_rate=learning_rate,
                             num_iters=num_iters)
    history_list.append(history)
    u_int = gcn(xy_int, A_int, d_int)
    # u_gcn = gcn(u_gcn, A_full, d_full)
    u_gcn = p_2d.assemble_sol(u_int.flatten()).reshape(-1,1)

    u_gcn = u_gcn.flatten()

    p_2d.plot_on_mesh(u_gcn,
                      f"GCN solution {num_points}",
                      f"{output_dir}{num_points}_u_gcn.png",
                      plot_with_lines=True,
                      )

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

    # Plot the loss history
    iter_ids = history["iter_ids"]
    loss_vals = history["loss_vals"]
    plt.plot(iter_ids, loss_vals, color="k")

    plt.xlabel("iter_id")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_gcn.png")
    plt.close()

    relative_l2_error = \
            jnp.linalg.norm(u_gcn - u_exct) / jnp.linalg.norm(u_exct)

    return relative_l2_error

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 4

    tt = main(num_points, output_dir = "trial/",
              num_iters=1000,
              learning_rate=1e-4)
    print(tt)

