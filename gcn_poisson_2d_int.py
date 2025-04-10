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
    # return x*y
    return 0.25*(1 - x*x - y*y)

def f(x,y):
    # return 0
    # return 2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return -2*(x**2 + y**2)
    # return 0
    return 1

def main(
            num_points,
            u = u,
            f = f,
            gcn_layers = [1, 10, 10, 1],
            num_fits = 10,
            iters_per_fit = 100,
            learning_rate = 5e-2,
            output_dir = "trial/"
            ):
    key_for_generating_random_numbers = jax.random.PRNGKey(69)

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
                      "",
                      f"{output_dir}{num_points}_u_exct.png")

# Gloabal stiffness matrix and consistent load vector
    K_mat, f_vec = p_2d.get_K_f()
    Kf = jnp.hstack((K_mat,f_vec.reshape(-1,1)))

# Adjacency matrix using the stiffness matrix
    A = jnp.zeros_like(K_mat, dtype=jnp.int8)
    A = A.at[jnp.nonzero(K_mat)].set(1)
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
    gcn = GCN(gcn_layers,
              [jnp.tanh] * (len(gcn_layers)-2) + [lambda x: x],
              model_key)

# Loss is typically a func of output and target
    def loss_fn(u, Kf):
        res = Kf[:,:-1] @ u - Kf[:,-1:]
        return jnp.sum(res*res)

# Training of the GCN model
    history_list = []
    for _ in range(num_fits):
        model = GCNModel(gcn, loss_fn)
        gcn, history = model.fit(u_gcn, A, deg, Kf,
                                 learning_rate=learning_rate,
                                 num_iters=iters_per_fit,
                                 num_check_points=iters_per_fit)
        history_list.append(history)
        u_gcn = gcn(u_gcn, A, deg)
    u_gcn = p_2d.assemble_sol(u_gcn.reshape(-1))

    p_2d.plot_on_mesh(u_gcn,
                      "",
                      f"{output_dir}{num_points}_u_gcn.png")
    p_2d.plot_on_mesh(u_gcn,
                      "",
                      f"{output_dir}{num_points}_u_gcn.png",
                      plot_with_lines = True)


    u_fem = p_2d.get_u_fem()
    p_2d.plot_on_mesh(u_fem,
                      "",
                      f"{output_dir}{num_points}_u_fem.png")

    p_2d.plot_on_mesh(jnp.abs(u_exct - u_fem),
                      "",
                      f"{output_dir}{num_points}_u_fem_err.png")

    p_2d.plot_on_mesh(jnp.abs(u_exct - u_gcn),
                      "",
                      f"{output_dir}{num_points}_u_gcn_err.png")

    # Plot the loss history
    for history_id,history in enumerate(history_list):
        iter_ids = history["iter_ids"]
        iter_ids += history_id*iters_per_fit
        loss_vals = history["loss_vals"]
        plt.plot(iter_ids, loss_vals, color="k")
        plt.axvline(x=iter_ids[0], color="r", linestyle='--', alpha=0.5)
    plt.xlabel("iteration number")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.savefig(f"{output_dir}{num_points}_loss_gcn_with_red_line.png")
    plt.close()
    for history_id,history in enumerate(history_list):
        iter_ids = history["iter_ids"]
        iter_ids += history_id*iters_per_fit
        loss_vals = history["loss_vals"]
        plt.plot(iter_ids, loss_vals, color="k")
    plt.xlabel("iteration number")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_gcn.png")
    plt.close()

    jnp.savez(f"{output_dir}details.npz",
              vertices = domain['vertices'],
              triangles = domain['triangles'],
              vertex_markers = domain['vertex_markers'],
              u_gcn = u_gcn,
              u_exct = u_exct,
              u_fem = u_fem,
              loss_vals = jnp.array([ h["loss_vals"] for h in history_list]),
              iter_ids = jnp.array([ h["iter_ids"] for h in history_list]),
              )

    relative_l2_error = \
            jnp.linalg.norm(u_gcn - u_exct) / jnp.linalg.norm(u_exct)

    return relative_l2_error, history_list

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 6

    plt.rcParams['font.size'] = 18
    plt.rcParams['savefig.bbox'] = 'tight'

    rel_l2_err, h_list = main(
            num_points,
            gcn_layers = [1] + [30]*2 + [1],
            iters_per_fit = 4000,
            num_fits = 1,
            learning_rate = 0.005,
            output_dir = "apr1/"
            )
