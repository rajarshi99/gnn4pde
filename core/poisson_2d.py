"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufactured soution using FEM
"""
import jax.numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from time import time

class Poisson_2d:

    def __init__(self, domain, u_bound_func, f_func):
        """
        Initialises methods to be used on the problem defined by the inputs.
        
        domain: dict with the following keys
            vertices: np.ndarray of shape (num of nodes, 2)
                each row with x,y coords of node_id(=row_id)
            vertex_markers: np.ndarray of shape (num of nodes, 1)
                entry 1 for if node_id(=row_id) is on the boundary; 0 otherwise
            triangles: np.ndarray of shape (num of elems, 3)
                each row with node ids of triangle
        u_bound_func: gives boundary value as output if input is a boundary point (x,y)
        f_func: forcing function gives output f(x,y)
        """
        self.vertex_markers = domain['vertex_markers'].reshape(-1)
        # self.vertex_known_id = self.vertex_markers.cumsum() - 1
        # self.vertex_unknown_id = (1 - self.vertex_markers).cumsum() - 1

        self.vert_known_list = np.where(self.vertex_markers == 1)[0].tolist()
        self.vert_unknown_list = np.where(self.vertex_markers == 0)[0].tolist()

        self.x = domain['vertices'][:,0]
        self.y = domain['vertices'][:,1]
        self.tri = domain['triangles']
        self.tri_flat = self.tri.flatten()

        self.triang = mtri.Triangulation(self.x, self.y, self.tri)  # to be used for plotting

        self.u_known = u_bound_func(self.x[self.vert_known_list], self.y[self.vert_known_list])
        self.n_known = len(self.vert_known_list)
        self.n_unknown = len(self.vert_unknown_list)
        self.n_elems = self.tri.shape[0]

        self.f = f_func

        self.u_sol = np.zeros(self.n_known + self.n_unknown)
        self.u_sol = self.u_sol.at[np.array(self.vert_known_list)].set(self.u_known)

        self.time_logs = []

    def clock_on(self, message):
        self.time_logs.append([time(), message])

    def clock_off(self):
        t_beg = self.time_logs[-1][0]
        self.time_logs[-1][0] = time() - t_beg
        
    def get_K_f(self):
        """
        Calculating global stiffness matrix and consistent load vector
        considering 3 node triangles, using TWEAKED lines of code from my own older code
        """
        self.clock_on("Forming K and f as dict")
        K_glob_dict = {}
        f_glob_dict = {}
        for e_id,vert_ids in enumerate(self.tri):
            x_vert = self.x[vert_ids]
            x02 = x_vert[0] - x_vert[2]
            x12 = x_vert[1] - x_vert[2]

            y_vert = self.y[vert_ids]
            y02 = y_vert[0] - y_vert[2]
            y12 = y_vert[1] - y_vert[2]

            detJ = x02*y12 - x12*y02
            B = np.array([[y12, -y02, -y12+y02],
                        [-x12, x02, x12-x02]]) / detJ
            K_elem = np.matmul(B.T,B) * detJ / 2

            x_mid = np.sum(x_vert) / 3
            y_mid = np.sum(y_vert) / 3
            f_mid = self.f(x_mid,y_mid)
            f_int = f_mid * detJ / 6

            for v_l_id,v_g_id in enumerate(vert_ids):
                if v_g_id in f_glob_dict:
                    f_glob_dict[v_g_id].append(f_int)
                else:
                    f_glob_dict[v_g_id] = [f_int]
                for u_l_id,u_g_id in enumerate(vert_ids):
                    if (v_g_id,u_g_id) in K_glob_dict:
                        K_glob_dict[(v_g_id,u_g_id)].append(K_elem[v_l_id,u_l_id])
                    else:
                        K_glob_dict[(v_g_id,u_g_id)] = [K_elem[v_l_id,u_l_id]]
        self.clock_off()

        self.clock_on("Forming K and f the global quantities")
        K_glob = np.zeros((self.n_unknown,self.n_unknown))
        K_known = np.zeros((self.n_unknown,self.n_known))
        f_force = np.zeros(self.n_unknown)

        # Explain and change names maybe
        K_glob_sparse = {
                "ind": [],
                "val": []
                }
        f_glob_sparse = {
                "ind": [],
                "val": [],
                }

        for v_ind, v_id in enumerate(self.vert_unknown_list):
            for u_ind, u_id in enumerate(self.vert_unknown_list):
                if (v_id,u_id) in K_glob_dict:
                    val = np.sum(np.array(K_glob_dict[(v_id,u_id)])).item()
                    K_glob = K_glob.at[v_ind,u_ind].set(val)
                    K_glob_sparse["ind"].append([v_id,u_id])
                    K_glob_sparse["val"].append(val)
            for u_ind, u_id in enumerate(self.vert_known_list):
                if (v_id,u_id) in K_glob_dict:
                    val = np.sum(np.array(K_glob_dict[(v_id,u_id)])).item()
                    K_known = K_known.at[v_ind,u_ind].set(val)
                    K_glob_sparse["ind"].append([v_id,u_id])
                    K_glob_sparse["val"].append(val)
            val = np.sum(np.array(f_glob_dict[v_id])).item()
            f_force = f_force.at[v_ind].set(val)
            f_glob_sparse["ind"].append(v_id)
            f_glob_sparse["val"].append(val)

        f_bound = np.dot(K_known,self.u_known)
        f_glob = f_force - f_bound
        self.clock_off()

        self.K_glob_dict = K_glob_dict
        self.f_glob_dict = f_glob_dict

        self.K_glob_sparse = K_glob_sparse
        self.f_glob_sparse = f_glob_sparse

        self.K_glob = K_glob
        self.f_glob = f_glob

        self.f_force = f_force
        self.f_bound = f_bound

        return K_glob, f_glob

    def get_K_f1_f2(self):
        """
        Returns K, f1 and f2 where
        K is the stiffness matrix ndof X ndof,
        f1 is the forcing vector
        f2 is the forcing due to Dirichlet boundary
        (So that we have K u = f1 - f2 to solve)
        """
        self.get_K_f()
        return self.K_glob, self.f_force, self.f_bound

    def get_u_fem(self):
        """
        Calls linear system solver and assembles u_sol
        """
        self.clock_on("Calling linear solver")
        u_unknown = np.linalg.solve(self.K_glob, self.f_glob)        
        self.clock_off()

        self.clock_on("Assembling u_sol")
        self.u_sol = self.u_sol.at[np.array(self.vert_unknown_list)].set(u_unknown)
        self.clock_off()

        self.u_unknown = u_unknown
        return self.u_sol

    def sol_FEM(self):
        self.get_K_f()
        self.u_sol = self.get_u_fem()
        return self.u_sol

    def assemble_sol(self, u_unknown_new):
        return self.u_sol.at[np.array(self.vert_unknown_list)].set(u_unknown_new)

    def K_norm(self, v):
        # Not sure if this is correct
        return v.dot(self.K_glob.dot(v))

    def L2_err(self, v):
        # Just a placeholder
        return np.linalg.norm(v)

    def l2_err(self, u_test):
        # Not a good measure of error?
        return np.linalg.norm(u_test - self.u_sol)

    def h_values(self):
        # A func to output the h value of each elem
        h_vals = np.zeros(self.n_elems)
        len_op3 = np.array([
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 1]])
        tri_pairs = [(0,1), (1,2), (2,0)]
        for e_id,vert_ids in enumerate(self.tri):
            x_coords = self.x[vert_ids]
            y_coords = self.y[vert_ids]
            x_lens = np.dot(len_op3, x_coords)
            y_lens = np.dot(len_op3, y_coords)
            lens = np.sqrt(x_lens*x_lens + y_lens*y_lens)
            h_vals[e_id] = np.max(lens)
        return h_vals

    def plot_on_mesh(self, u_inp, title = " ", fname = None, plot_with_lines = False):
        cplot = plt.tricontourf(self.triang, u_inp, cmap = "jet", levels = 200)
        plt.colorbar(cplot)
        if plot_with_lines:
            plt.triplot(self.triang, 'ko-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)

        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)
            plt.close()

    def plot_sol_on_mesh(self, title = " ", fname = False, plot_with_lines = True):
        # This line looks a little sad
        self.plot_on_mesh(self.u_sol, title = title, fname = fname, plot_with_lines = plot_with_lines)
        
