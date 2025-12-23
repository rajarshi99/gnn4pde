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
    return x*y

def f(x,y):
    return 0

def sdf(x,y):
    dx = jnp.abs(x) - 1
    dy = jnp.abs(y) - 1

    return jnp.minimum(jnp.maximum(dx,dy),0) + jnp.sqrt(jnp.maximum(dx,0)**2 + jnp.maximum(dy,0)**2)

class GNIN(GCN):
    """
    Graph Neural Interaction Network
    """
    fno : FNO2d
    N_lat_x : int
    N_lat_y : int

    def __init__(
            self,
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
        fno_out_mat = self.fno(fno_inp_mat)
        return jnp.sum((V @ fno_out_mat) * W, axis=1, keepdims=True)

def main(
            num_points,
            u = u,
            f = f,
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
    xys_inp = jnp.column_stack((x, y, sdf(x,y)))

    # EXPLAIN
    model_key, key_for_generating_random_numbers = \
            jax.random.split(key_for_generating_random_numbers)
    gnin = GNIN(
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

    # Loss is typically a func of output and target
    def loss_fn(u, Kf):
        u = u.reshape(-1,1) # Change later
        print(u.shape, Kf.shape)
        res = Kf[:,:-1] @ u - Kf[:,-1:]
        return jnp.sum(res*res)

    return xys_inp, A, deg, gnin, loss_fn

"""
    # Training of the FNO
    model = GCNModel(gnin, loss_fn) #, [rel_l2_err_fn, f_val_fn, penalty_fun])
    gnin, history = model.fit(xys_inp, A, deg, Kf,
                             learning_rate=learning_rate,
                             num_iters=num_iters,
                             num_check_points=num_iters)
    u_gcn = gnin(xys_inp, A, deg)

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
