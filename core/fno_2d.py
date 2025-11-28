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
Module: fno_2d.py

This module provides the implementation of the two-dimensional Fourier Neural Operator (FNO) model.

Classes:
    FNO2d: 2D Fourier Neural Operator

Dependencies:
    - jax: For array processing
    - equinox: For neural network layers

Key Features:
    - Lifting layer
    - FNO blocks
    - Projection layer

Authors:
    Diya Nag Chaudhury

Version Info:
    28/Jan/2025: Initial version - Diya Nag Chaudhury

References:
    None
"""
import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from tqdm import tqdm

from core.fno_block_2d import FNOBlock2d

class FNO2d(eqx.Module):
    """Complete 2D Fourier Neural Operator

    This module combines the lifting layer, FNO blocks, and projection layer
    to create a complete 2D Fourier Neural Operator.

    Attributes:
    lifting: eqx.nn.Conv2d
    fno_blocks: list
    projection: eqx.nn.Conv2d

    Methods:
    __init__: Initializes the FNO2d object
    __call__: Calls the FNO2d object
    """

    lifting: eqx.nn.Conv2d
    fno_blocks: list
    projection: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        width,
        activation,
        n_blocks,
        *,
        key,
    ):
        """
        Initializes the FNO2d object

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            modes1 (_type_): _description_
            modes2 (_type_): _description_
            width (_type_): _description_
            activation (_type_): _description_
            n_blocks (_type_): _description_
            key (_type_): _description_
        """

        keys = jax.random.split(key, n_blocks + 2)

        self.lifting = eqx.nn.Conv2d(in_channels, width, kernel_size=1, key=keys[0])

        self.fno_blocks = []
        for i in range(n_blocks):
            self.fno_blocks.append(
                FNOBlock2d(width, width, modes1, modes2, activation, key=keys[i + 1])
            )

        self.projection = eqx.nn.Conv2d(
            width, out_channels, kernel_size=1, key=keys[-1]
        )

    def __call__(self, x):
        """_
        Calls the FNO2d object

        Args:
        x: jnp.ndarray

        Returns:
        jnp.ndarray
        """
        x = self.lifting(x)

        for block in self.fno_blocks:
            x = block(x)

        x = self.projection(x)
        return x

class FNO2dModel:

    def __init__(
        self,
        fno: FNO2d,
        loss_fn: callable,
        metrics: list[callable] = [],
    ):
        """
        Initialize the fno model with network architecture and training parameters.

        Args:
            fno: Neural network architecture to train.
            loss_fn (Callable): Loss function for training.
            metrics (list[Callable]): List of metric functions for evaluation.
        """

        self.fno = fno
        self.loss_fn = loss_fn
        self.metrics = metrics

    def fit(
        self,
        features: jnp.ndarray,
        target,
        learning_rate: float,
        num_iters: int = 10,
        num_check_points: int = 5,
    ):
        """
        Train the fno

        Args:
            features: jnp.ndarray,
            target,
            learning_rate: Parameter of the gradient based optimisation method
            num_iters: Number of iterations of the gradient based optimisation method
            num_check_points

        Returns:
            Trained fno
        """
        fno = self.fno
        history = {
                "iter_ids": jnp.zeros(num_check_points),
                "loss_vals": jnp.zeros(num_check_points),
                "metric_vals":
                jnp.zeros((num_check_points,len(self.metrics)))
                }
        history_id = 0

        self.optimizer = optax.adam(learning_rate)
        opt_state = self.optimizer.init(eqx.filter(fno, eqx.is_array))

        check_point_gap = num_iters / num_check_points

        # for iter_id in range(num_iters):
        for iter_id in tqdm(range(num_iters), desc="Training", total=num_iters):

            loss, fno, opt_state = self._update_step(
                fno, features, target, opt_state
            )

            if iter_id % check_point_gap == 0:
                output = fno(features)
                metric_vals = [m(output) for m in self.metrics]
                history["iter_ids"] = history["iter_ids"].at[history_id].set(iter_id)
                history["loss_vals"] = history["loss_vals"].at[history_id].set(loss)

                history["metric_vals"] = history["metric_vals"].at[history_id,:].set(metric_vals)
                history_id += 1
                # print(f"Iter: {iter_id} | Loss: {loss:.2e} | Metrics {metric_vals}")

        return fno, history

    @eqx.filter_jit
    def _update_step(
        self,
        fno: FNO2d,
        features: jnp.ndarray,
        target,
        opt_state,
    ):
        """
        Perform single training step with JIT compilation.

        Args:
            fno: FNO2d,
            features: Input feature vectors
            target
            opt_state

        Returns:
            tuple: Tuple containing:
                - Average loss for the batch
                - Updated network
                - Updated optimizer state
        """
        loss, grads = eqx.filter_value_and_grad(self._loss_fn)(
            fno, features, target
        )

        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(fno, eqx.is_array)
        )
        fno = eqx.apply_updates(fno, updates)

        return loss, fno, opt_state

    def _loss_fn(
        self,
        fno: FNO2d,
        features: jnp.ndarray,
        target,
    ):
        """
        Compute loss for the given input data.
        Required for getting gradients during training and JIT.

        Args:
            fno: Instance of FNO2d
            features: Input feature vectors
            target

        Returns:
            jnp.ndarray: Loss value.
        """
        return self.loss_fn(
            fno(features), target
        ).mean()

