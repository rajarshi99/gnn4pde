"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufacture soution using GCN "+" FNO
where the inputs are the coordinates and th SDF
for each node and G = (V,E)

Here we define
- u (unknown function)
- f (forcing function)
- x and y the coordinates of the points
"""

import jax
import jax.numpy as jnp

from core.gcn import GCN, GCNModel
from core.fno_2d import FNO2d
from core.poisson_2d import Poisson_2d

import triangle as tr
import sys

import matplotlib.pyplot as plt

def u(x,y):
    return 1.2*0.25*(1 - x*x - y*y)

def f(x,y):
    return 1.2

def f_guess(x,y):
    return 1

def sdf(x,y):
    dx = jnp.abs(x) - 1
    dy = jnp.abs(y) - 1

    return jnp.minimum(jnp.maximum(dx,dy),0) + jnp.sqrt(jnp.maximum(dx,0)**2 + jnp.maximum(dy,0)**2)

class GNINGuessForcing(GCN):
    """
    Graph Neural Interaction Network : GCN "+" FNO
    with trainable uniform forcing to be found
    """
    fno : FNO2d
    N_lat_x : int
    N_lat_y : int

    f_val: jnp.ndarray

    def __init__(
            self,
            f_guess_val,
            gcn_hidden_layers,
            gcn_hidden_activations,
            fno_in_channels,
            fno_out_channels,
            fno_modes1,
            fno_modes2,
            fno_width,
            fno_activation,
            fno_n_blocks,
            N_lat_x,
            N_lat_y,
            key,
            ):
        self.f_val = jnp.array(f_guess_val)
        gcn_key, fno_key = jax.random.split(key, 2)
        super().__init__([3] + gcn_hidden_layers + [N_lat_x + N_lat_y],
                         [lambda x: x] + gcn_hidden_activations + [lambda x :x],
                         gcn_key)
        self.fno = FNO2d(
            in_channels=fno_in_channels,
            out_channels=fno_out_channels,
            modes1=fno_modes1,
            modes2=fno_modes2,
            width=fno_width,
            activation=fno_activation,
            n_blocks=fno_n_blocks,
            key=fno_key,
            )
        self.N_lat_x = N_lat_x
        self.N_lat_y = N_lat_y

    def __call__(self, X, adj_mat, degree):
        gcn_out = super().__call__(X, adj_mat, degree)
        V = gcn_out[:,:self.N_lat_x] # Shape (N_in,N_lat_x)
        W = gcn_out[:,self.N_lat_x:] # Shape (N_in,N_lat_y)
        fno_inp_mat = V.T @ W
        fno_out_mat = self.fno(fno_inp_mat.reshape(1,self.N_lat_x,self.N_lat_y))
        return jnp.sum((V @ fno_out_mat[0]) * W, axis=1), self.f_val

def main(
            num_points,
            u = u,
            f = f_guess,
            gcn_num_hidden_layers = 3,
            gcn_num_hidden_neurons = 10,
            N_lat_x = 16,
            N_lat_y = 16,
            fno_modes1 = 5,
            fno_modes2 = 5,
            fno_width  = 1,
            fno_activation = jnp.tanh,
            fno_n_blocks = 2,
            num_iters = 100,
            learning_rate = 5e-2,
            num_internal_data_points = 2,
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
                      "",
                      f"{output_dir}{num_points}_u_exct.png")

    # EXPLAIN
    K_mat, f_force, f_bound = p_2d.get_K_f1_f2()
    Kf1f2 = [K_mat, f_force.reshape(-1,1), f_bound.reshape(-1,1)]

    # Adjacency matrix for interior points
    A = jnp.zeros_like(K_mat, dtype=jnp.int8)
    A = A.at[jnp.nonzero(K_mat)].set(1)
    deg = A.sum(axis=0)

    # Input are the xy coordinates and SDF values the shape is num_unknowns X 3
    vert_unknown_list = jnp.array(p_2d.vert_unknown_list)
    x = x.at[vert_unknown_list].get()
    y = y.at[vert_unknown_list].get()
    xys_inp = jnp.column_stack((x, y, sdf(x,y)))

    # EXPLAIN
    model_key, key_for_generating_random_numbers = \
            jax.random.split(key_for_generating_random_numbers)
    gnin = GNINGuessForcing(
            f_guess_val = 1.0,
            gcn_hidden_layers = [gcn_num_hidden_neurons] * gcn_num_hidden_layers,
            gcn_hidden_activations = [jnp.tanh] * gcn_num_hidden_layers,
            fno_in_channels = 1,
            fno_out_channels = 1,
            fno_modes1 = fno_modes1,
            fno_modes2 = fno_modes2,
            fno_width = fno_width,
            fno_activation = fno_activation,
            fno_n_blocks = fno_n_blocks,
            N_lat_x = N_lat_x,
            N_lat_y = N_lat_y,
            key = model_key,
            )

    data_node_selection_key, key_for_generating_random_numbers = jax.random.split(key_for_generating_random_numbers)
    data_node_inds = jax.random.permutation(data_node_selection_key, jnp.arange(vert_unknown_list.shape[0]))[:num_internal_data_points]
    data_node_u_vals = u(xys_inp[data_node_inds,0], xys_inp[data_node_inds,1])

    def penalty_fun(output):
        u = output[0]
        penalty = u[data_node_inds] - data_node_u_vals
        return jnp.sum(penalty*penalty)

    # Loss is typically a func of output and target
    def loss_fn(output, Kf1f2):
        K_mat, f_force, f_data = Kf1f2
        u, f_val = output
        # f_val = 1.2
        res = (K_mat @ u) - (f_val * f_force) + f_data
        penalty = u[data_node_inds] - data_node_u_vals
        penalty = jnp.sum(penalty*penalty)/penalty.shape[0]
        return jnp.sum(res*res)/res.shape[0] + 1e1*penalty

    u_exct_int = u_exct.at[vert_unknown_list].get()

    def rel_l2_err_fn(output):
        u = output[0]
        err = u.flatten() - u_exct_int.flatten()
        return jnp.linalg.norm(err) / jnp.linalg.norm(u_exct)

    def f_val_fn(output):
        return output[1]

    # Training the GNIN
    model = GCNModel(gnin, loss_fn, [rel_l2_err_fn, f_val_fn, penalty_fun])
    gnin, history = model.fit(xys_inp, A, deg, Kf1f2,
                             learning_rate=learning_rate,
                             num_iters=num_iters,
                             num_check_points=num_iters)
    u_gnin = gnin(xys_inp, A, deg)

    u_gnin = p_2d.assemble_sol(u_gnin[0].reshape(-1))

    p_2d.plot_on_mesh(u_gnin,
                      "",
                      f"{output_dir}{num_points}_u_gnin.png")
    p_2d.plot_on_mesh(u_gnin,
                      "",
                      f"{output_dir}{num_points}_u_gnin(2).png",
                      plot_with_lines = True)

    u_fem = p_2d.get_u_fem()
    p_2d.plot_on_mesh(u_fem,
                      "",
                      f"{output_dir}{num_points}_u_fem.png")

    p_2d.plot_on_mesh(jnp.abs(u_exct - u_fem),
                      "",
                      f"{output_dir}{num_points}_u_fem_err.png")

    p_2d.plot_on_mesh(jnp.abs(u_exct - u_gnin),
                      "",
                      f"{output_dir}{num_points}_u_gnin_err.png")

    # The commented below part causes issues...
    # plt = p_2d.get_mesh_plt(jnp.abs(u_exct - u_gnin))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.scatter(xys_inp[data_node_inds,0], xys_inp[data_node_inds,1],
    #             marker = "o", color = "black")
    # plt.savefig(f"{output_dir}{num_points}_u_gcn_err_int_data.png")
    # plt.close()

    # Plot the loss history
    iter_ids = history["iter_ids"]
    loss_vals = history["loss_vals"]
    plt.plot(iter_ids, loss_vals, color="k")
    plt.xlabel("iteration number")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.savefig(f"{output_dir}{num_points}_loss_gnin.png")
    plt.close()

    # Scatter plot of loss vs error
    rel_l2_err_vals = history["metric_vals"][:,0]
    plt.scatter(loss_vals, rel_l2_err_vals, c=iter_ids, cmap='jet')
    plt.xlabel("Loss")
    plt.ylabel("Relative l2 loss")
    plt.colorbar()
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_loss_err.png")
    plt.close()

    # Plot the f vals
    f_vals = history["metric_vals"][:,1]
    plt.plot(iter_ids, f_vals, color="r")
    plt.xlabel("iteration number")
    plt.ylabel("f")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_f_val.png")
    plt.close()

    plt.scatter(loss_vals, rel_l2_err_vals, c=iter_ids, cmap='jet')
    plt.xlabel("Loss")
    plt.ylabel("Relative l2 loss")
    plt.colorbar()
    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(f"{output_dir}{num_points}_loss_err_log.png")
    plt.close()


    penalty_vals = history["metric_vals"][:,2]
    plt.plot(iter_ids, penalty_vals, color="black")
    plt.xlabel("iteration number")
    plt.ylabel("penalty value")
    plt.yscale("log")
    plt.grid()
    plt.savefig(f"{output_dir}{num_points}_penalty_val.png")
    plt.close()

    jnp.savez(f"{output_dir}details.npz",
              vertices = domain['vertices'],
              triangles = domain['triangles'],
              vertex_markers = domain['vertex_markers'],
              u_gnin = u_gnin,
              u_exct = u_exct,
              u_fem = u_fem,
              loss_vals = loss_vals,
              rel_l2_err_vals = rel_l2_err_vals,
              metric_vals = history["metric_vals"]
              )

    relative_l2_error = \
            jnp.linalg.norm(u_gnin - u_exct) / jnp.linalg.norm(u_exct)

    return relative_l2_error

if __name__ == "__main__":
    try:
        num_points = int(sys.argv[1])
    except:
        num_points = 4

    err = main(num_points, u, f_guess, num_iters=10000, learning_rate=5e-4, output_dir = "trial/")
