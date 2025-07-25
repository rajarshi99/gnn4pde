# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""
    Module: gcn.py

    This module implements Graph Convolution Networks (GCNs)

    It provides functionality to:
        - Perform transformations on Graphs using GCN
        - Train GCNs for a loss function

    Key Features:
        - Built on top of base class getting all its functionalities
        - Efficient neural networks implementation using equinox modules

    Authors:
        - Rajarshi Dasgupta (rajarshid@iisc.ac.in)

    Version Info:
        - 10/01/2025: Initial version

"""
import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from tqdm import tqdm


class GCN(eqx.Module):
    num_layers: int
    W_list: list
    B_list: list

    activations: list

    def __init__(self, layers, activations, key):
        """
        Initialize a GCN instance with random initial parameters

        Inputs:
            layers: a python list indicating the size of the node embeddings at each layer
            activations: a python list of activation functions
            key: to generate random numbers for initialising the W and B matrices
        """

        self.num_layers = len(layers)
        self.W_list = []
        self.B_list = []

        self.activations = activations

        for i in range(self.num_layers - 1):
            weights_key, bias_key, key = jax.random.split(key, num=3)
            W = jax.random.normal(weights_key, (layers[i], layers[i + 1]))
            B = jax.random.normal(bias_key, (layers[i], layers[i + 1]))

            self.W_list.append(W)
            self.B_list.append(B)

    def __call__(self, z, adj_mat, degree):
        """
        Initialize the gcn model with network architecture and training parameters.

        Args:
            z: jnp array for which the i-th row is the i-th node embedding
            adj_mat: the adjacency matrix. Ideally it should be a sparse matrix
            degree: jnp array where the i-th element is the degree of the i-th node

        Output:
            node embeddings of the output
        """

        # activation = jnp.tanh
        # for W,B in zip(self.W_list,self.B_list):
        for activation, W, B in zip(self.activations, self.W_list, self.B_list):
            z = activation(jnp.diagflat(1.0 / degree) @ adj_mat @ z @ W + z @ B)
        return z


class GCNModel:

    def __init__(
        self,
        gcn: GCN,
        loss_fn: callable,
        metrics: list[callable] = [],
    ):
        """
        Initialize the gcn model with network architecture and training parameters.

        Args:
            gcn: Neural network architecture to train.
            loss_fn (Callable): Loss function for training.
            metrics (list[Callable]): List of metric functions for evaluation.
        """

        self.gcn = gcn
        self.loss_fn = loss_fn
        self.metrics = metrics

    def fit(
        self,
        features: jnp.ndarray,
        adjacency_matrix: jnp.ndarray,
        degree_array: jnp.ndarray,
        target,
        learning_rate: float,
        num_iters: int = 10,
        num_check_points: int = 5,
    ):
        """
        Train the gcn

        Args:
            features: jnp.ndarray,
            adjacency_matrix: jnp.ndarray,
            degree_array: jnp.ndarray,
            target,
            learning_rate: Parameter of the gradient based optimisation method
            num_iters: Number of iterations of the gradient based optimisation method
            num_check_points

        Returns:
            Trained gcn
        """
        gcn = self.gcn
        history = {
                "iter_ids": jnp.zeros(num_check_points),
                "loss_vals": jnp.zeros(num_check_points),
                "metric_vals":
                jnp.zeros((num_check_points,len(self.metrics)))
                }
        history_id = 0

        self.optimizer = optax.adam(learning_rate)
        opt_state = self.optimizer.init(eqx.filter(gcn, eqx.is_array))

        check_point_gap = num_iters / num_check_points

        # for iter_id in range(num_iters):
        for iter_id in tqdm(range(num_iters), desc="Training", total=num_iters):

            loss, gcn, opt_state = self._update_step(
                gcn, features, adjacency_matrix, degree_array, target, opt_state
            )

            if iter_id % check_point_gap == 0:
                output = gcn(features, adjacency_matrix, degree_array)
                metric_vals = [m(output) for m in self.metrics]
                history["iter_ids"] = history["iter_ids"].at[history_id].set(iter_id)
                history["loss_vals"] = history["loss_vals"].at[history_id].set(loss)

                history["metric_vals"] = history["metric_vals"].at[history_id,:].set(metric_vals)
                history_id += 1
                # print(f"Iter: {iter_id} | Loss: {loss:.2e} | Metrics {metric_vals}")

        return gcn, history

    @eqx.filter_jit
    def _update_step(
        self,
        gcn: GCN,
        features: jnp.ndarray,
        adjacency_matrix: jnp.ndarray,
        degree_array: jnp.ndarray,
        target,
        opt_state,
    ):
        """
        Perform single training step with JIT compilation.

        Args:
            gcn: GCN,
            features: Input feature vectors
            adjacency_matrix
            degree_array
            target
            opt_state

        Returns:
            tuple: Tuple containing:
                - Average loss for the batch
                - Updated network
                - Updated optimizer state
        """
        loss, grads = eqx.filter_value_and_grad(self._loss_fn)(
            gcn, features, adjacency_matrix, degree_array, target
        )

        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(gcn, eqx.is_array)
        )
        gcn = eqx.apply_updates(gcn, updates)

        return loss, gcn, opt_state

    def _loss_fn(
        self,
        gcn: GCN,
        features: jnp.ndarray,
        adjacency_matrix,
        degree_array,
        target,
    ):
        """
        Compute loss for the given input data.
        Required for getting gradients during training and JIT.

        Args:
            gcn: Instance of GCN
            features: Input feature vectors
            adjacency_matrix,
            degree_array,
            target

        Returns:
            jnp.ndarray: Loss value.
        """
        return self.loss_fn(
            gcn(features, adjacency_matrix, degree_array), target
        ).mean()

class ChebyshevGCN(eqx.Module):
    """
    PI-GGN Style GCN
    """
    num_layers: int
    poly_order: int
    theta_list: list
    b_list: list

    activations: list

    def __init__(self, layers, activations, num_nodes, key, poly_order = 10):
        """
        Initialize a Chebyshev GCN instance with random initial parameters
        and trainable bias vector for fixed num of nodes graph

        Inputs:
            layers: a python list indicating the size of the node embeddings at each layer
            activations: a python list of activation functions
            key: to generate random numbers for initialising the W and B matrices
        """

        self.num_layers = len(layers)
        self.poly_order = poly_order
        self.theta_list = []
        self.b_list = []

        self.activations = activations

        for i in range(self.num_layers - 1):
            b_key, W_key, key = jax.random.split(key, num=3)
            W_key = jax.random.split(W_key, num=poly_order)

            b = jax.random.normal(b_key, (num_nodes, layers[i + 1]))

            W_list = []
            for k in range(poly_order):
                W = jax.random.normal(W_key[k], (layers[i], layers[i + 1]))
                W_list.append(W)

            self.theta_list.append(W_list)
            self.b_list.append(b)

    def __call__(self, X, adj_mat, degree):
        """
        Initialize the gcn model with network architecture and training parameters.

        Args:
            z: jnp array for which the i-th row is the i-th node embedding
            adj_mat: the adjacency matrix. Ideally it should be a sparse matrix
            degree: jnp array where the i-th element is the degree of the i-th node

        Output:
            node embeddings of the output
        """

        # activation = jnp.tanh
        # for W,B in zip(self.W_list,self.B_list):
        lap_hat_mat = - jnp.diagflat(1.0 / degree) @ adj_mat
        for activation, W_list, b in zip(self.activations, self.theta_list, self.b_list):
            Z_list = [X, lap_hat_mat @ X]
            for k in range(2, self.poly_order):
                Z_list.append((2 * lap_hat_mat @ Z_list[k-1]) - Z_list[k-2])
            X = activation(jnp.sum(jnp.stack([Z @ W for Z,W in zip(Z_list,W_list)]), axis=0) + b)
        return X
