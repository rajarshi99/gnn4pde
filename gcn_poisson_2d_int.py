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
import time

omegax = 2*jnp.pi
omegay = 2*jnp.pi
r1 = 10

def u(x,y):
    # return jnp.where(y == 1, 1, 0)
    # return jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return x**2*y**2
    # return x*y
    # return 0.25*(1 - x*x - y*y)
    return (0.1*jnp.sin(omegax*x) + jnp.tanh(r1*x)) * jnp.sin(omegay*(y))

def f(x,y):
    # return 0
    # return 2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return -2*(x**2 + y**2)
    # return 0
    # return 1
    gtemp = (-0.1*(omegax**2)*jnp.sin(omegax*x) - (2*r1**2)*(jnp.tanh(r1*x))/((jnp.cosh(r1*x))**2))*jnp.sin(omegay*(y))\
            +(0.1*jnp.sin(omegax*x) + jnp.tanh(r1*x)) * (-omegay**2 * jnp.sin(omegay*(y)) )
    return -gtemp

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

    vert_unknown_list = jnp.array(p_2d.vert_unknown_list)
    u_exct_int = u_exct.at[vert_unknown_list].get()

    def rel_l2_err_fn(u):
        err = u.flatten() - u_exct_int.flatten()
        return jnp.linalg.norm(err) / jnp.linalg.norm(u_exct)

# Training of the GCN model
    history_list = []
    start_time = time.time()
    for _ in range(num_fits):
        model = GCNModel(gcn, loss_fn, [rel_l2_err_fn])
        gcn, history = model.fit(u_gcn, A, deg, Kf,
                                 learning_rate=learning_rate,
                                 num_iters=iters_per_fit,
                                 num_check_points=iters_per_fit)
        history_list.append(history)
        u_gcn = gcn(u_gcn, A, deg)
    elapsed_time = time.time() - start_time
    print("Training time: ", elapsed_time)

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

    for history_id,history in enumerate(history_list):
        iter_ids = history["iter_ids"]
        iter_ids += history_id*iters_per_fit
        loss_vals = history["loss_vals"]
        rel_l2_err_vals = history["metric_vals"][:,0]
        plt.scatter(loss_vals, rel_l2_err_vals,
                    c=iter_ids, cmap='jet')
    plt.xlabel("Loss")
    plt.ylabel("Relative l2 loss")
    plt.colorbar()
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_err.png")
    plt.close()

    for history_id,history in enumerate(history_list):
        iter_ids = history["iter_ids"]
        iter_ids += history_id*iters_per_fit
        loss_vals = history["loss_vals"]
        rel_l2_err_vals = history["metric_vals"][:,0]
        plt.scatter(loss_vals, rel_l2_err_vals,
                    c=iter_ids, cmap='jet')
    plt.xlabel("Loss")
    plt.ylabel("Relative l2 loss")
    plt.colorbar()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(f"{output_dir}{num_points}_loss_err_log.png")
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

    gcn_relative_l2_error = \
            jnp.linalg.norm(u_gcn - u_exct) / jnp.linalg.norm(u_exct)
    gcn_l_inf_error = jnp.max(jnp.abs(u_gcn - u_exct))

    fem_relative_l2_error = \
            jnp.linalg.norm(u_fem - u_exct) / jnp.linalg.norm(u_exct)
    fem_l_inf_error = jnp.max(jnp.abs(u_fem - u_exct))

    print("gcn rel l2 error: ", gcn_relative_l2_error)
    print("gcn l inf error: ", gcn_l_inf_error)

    fem_relative_l2_error = \
            jnp.linalg.norm(u_fem - u_exct) / jnp.linalg.norm(u_exct)
    fem_l_inf_error = jnp.max(jnp.abs(u_fem - u_exct))

    print("fem rel l2 error: ", fem_relative_l2_error)
    print("fem l inf error: ", fem_l_inf_error)

    return history_list

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 12

    plt.rcParams['font.size'] = 18
    plt.rcParams['savefig.bbox'] = 'tight'

    h_list = main(
            num_points,
            gcn_layers = [1] + [40]*3 + [1],
            iters_per_fit = 250000,
            # iters_per_fit = 9500,
            num_fits = 1,
            learning_rate = 0.0001,
            output_dir = "may4/"
            )
