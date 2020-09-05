import jax
import jax.numpy as np
from jax import random
from jax.nn.initializers import orthogonal


def RandomRotations():
    """An implementation of an invertible linear layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        random_matrix = orthogonal()(rng, (input_dim, input_dim))
        q, _ = jax.scipy.linalg.qr(random_matrix)
        random_matrix = q
        ldj = np.log(np.abs(random_matrix)).sum()

        def direct_fun(params, inputs, **kwargs):

            outputs = inputs @ random_matrix
            log_det_jacobian = np.full(inputs.shape[:1], ldj)

            # print(outputs.shape, log_det_jacobian.shape)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):

            outputs = inputs @ random_matrix.T
            log_det_jacobian = np.full(inputs.shape[:1], -ldj)
            return outputs, log_det_jacobian

        return (), direct_fun, inverse_fun

    return init_fun
