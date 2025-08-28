"""
Inverse Poisson problem: -laplacian(u)(x,y) = f
Find u(x,y) and unknown constant f given one extra data point
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
    return 1.2*0.25*(1 - x*x - y*y)
    # return (0.1*jnp.sin(omegax*x) + jnp.tanh(r1*x)) * jnp.sin(omegay*(y))

def f(x,y):
    # return 0
    # return 2*(jnp.pi)**2*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)
    # return -2*(x**2 + y**2)
    # return 0
    return 1.2
    # gtemp = (-0.1*(omegax**2)*jnp.sin(omegax*x) - (2*r1**2)*(jnp.tanh(r1*x))/((jnp.cosh(r1*x))**2))*jnp.sin(omegay*(y))\
            # +(0.1*jnp.sin(omegax*x) + jnp.tanh(r1*x)) * (-omegay**2 * jnp.sin(omegay*(y)) )
    # return -gtemp

def f_guess(x,y):
    return 1

class GCNGuessForcing(GCN):
    """
    An extra trainable parameter over the parent class
    """
    f_val: jnp.ndarray

    def __init__(self, layers, activations, key, f_val = 1.0):
        super().__init__(layers, activations, key)
        self.f_val = jnp.array(f_val)

    def __call__(self, X, adj_mat, degree):
        gcn_out = super().__call__(X, adj_mat, degree)
        return gcn_out, self.f_val


def main(
            num_points,
            u = u,
            f_guess = f_guess,
            gcn_layers = [1, 10, 10, 1],
            num_iters = 10,
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

    p_2d = Poisson_2d(domain, u, f_guess)

    u_exct = u(x,y)
    p_2d.plot_on_mesh(u_exct,
                      "",
                      f"{output_dir}{num_points}_u_exct.png")

# Gloabal stiffness matrix and consistent load vector
    K_mat, f_force, f_bound = p_2d.get_K_f1_f2()
    Kf1f2 = [K_mat, f_force.reshape(-1,1), f_bound.reshape(-1,1)]

# Adjacency matrix using the stiffness matrix
    A = jnp.zeros_like(K_mat, dtype=jnp.int8)
    A = A.at[jnp.nonzero(K_mat)].set(1)
    deg = A.sum(axis=0)

# Input is the xy coords at the degree of freedom
    n_unknown = p_2d.n_unknown
    dof_inds = jnp.array(p_2d.vert_unknown_list)
    xy_inp = jnp.column_stack((x[dof_inds],y[dof_inds]))
    init_key, key_for_generating_random_numbers = jax.random.split(key_for_generating_random_numbers)
    u_gcn = jax.random.normal(init_key, (n_unknown,1))

    data_node_inds = jnp.array([int(dof_inds.shape[0]/2)])
    data_node_u_vals = u(xy_inp[data_node_inds,0], xy_inp[data_node_inds,1])

    print("Internal data node indices", data_node_inds)
    print("Internal data node u vals", data_node_u_vals)

# GCN with 2 hidden layers,
# Last activation func as identity
# Other activations are tanh
    model_key, key_for_generating_random_numbers = \
            jax.random.split(key_for_generating_random_numbers)
    gcn = GCNGuessForcing(gcn_layers,
                       [jnp.tanh] * (len(gcn_layers)-2) + [lambda x: x],
                       key = model_key)

# Loss is typically a func of output and target
    def penalty_fun(output):
        u = output[0]
        penalty = u[data_node_inds] - data_node_u_vals
        return jnp.sum(penalty*penalty)

    def loss_fn(output, Kf1f2):
        K_mat, f_force, f_data = Kf1f2
        u, f_val = output
        # f_val = 1.2
        res = (K_mat @ u) - (f_val * f_force) + f_data
        penalty = u[data_node_inds] - data_node_u_vals
        penalty = jnp.sum(penalty*penalty)
        return jnp.sum(res*res) + penalty

    vert_unknown_list = jnp.array(p_2d.vert_unknown_list)
    u_exct_int = u_exct.at[vert_unknown_list].get()

    def rel_l2_err_fn(output):
        u = output[0]
        err = u.flatten() - u_exct_int.flatten()
        return jnp.linalg.norm(err) / jnp.linalg.norm(u_exct)

    def f_val_fn(output):
        return output[1]

# Training of the GCN model
    start_time = time.time()
    model = GCNModel(gcn, loss_fn, [rel_l2_err_fn, f_val_fn, penalty_fun])
    gcn, history = model.fit(u_gcn, A, deg, Kf1f2,
                             learning_rate=learning_rate,
                             num_iters=num_iters,
                             num_check_points=num_iters)
    gcn_output = gcn(u_gcn, A, deg)
    u_gcn = gcn_output[0]
    elapsed_time = time.time() - start_time
    print("Training time: ", elapsed_time)

    u_gcn = p_2d.assemble_sol(u_gcn.reshape(-1))
    f_final = gcn.f_val
    print("Final f value obtained", f_final, gcn_output[1])

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

    plt = p_2d.get_mesh_plt(jnp.abs(u_exct - u_gcn))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(xy_inp[data_node_inds,0], xy_inp[data_node_inds,1],
                marker = "o", color = "black")
    plt.savefig(f"{output_dir}{num_points}_u_gcn_err.png")
    plt.close()

    # Plot the loss history
    iter_ids = history["iter_ids"]
    loss_vals = history["loss_vals"]
    plt.plot(iter_ids, loss_vals, color="k")
    plt.xlabel("iteration number")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_gcn.png")
    plt.close()

    f_vals = history["metric_vals"][:,1]
    plt.plot(iter_ids, f_vals, color="r")
    plt.xlabel("iteration number")
    plt.ylabel("f")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_f_val.png")
    plt.close()

    penalty_vals = history["metric_vals"][:,2]
    plt.plot(iter_ids, penalty_vals, color="r")
    plt.xlabel("iteration number")
    plt.ylabel("penalty value")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_penalty_val.png")
    plt.close()

    loss_vals = history["loss_vals"]
    rel_l2_err_vals = history["metric_vals"][:,0]
    plt.scatter(loss_vals, rel_l2_err_vals, c=iter_ids, cmap='jet')
    plt.xlabel("Loss")
    plt.ylabel("Relative l2 loss")
    plt.colorbar()
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_err.png")
    plt.close()

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
              loss_vals = loss_vals,
              iter_ids = iter_ids,
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

    return xy_inp, Kf1f2

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 12

    plt.rcParams['font.size'] = 18
    plt.rcParams['savefig.bbox'] = 'tight'

    xy_inp, Kf1f2 = main(
            num_points,
            gcn_layers = [1] + [30]*4 + [1],
            # num_iters = 500,
            num_iters = 25000,
            learning_rate = 0.0005,
            output_dir = "output/"
            )
