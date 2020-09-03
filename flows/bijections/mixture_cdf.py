import jax
from jax import random
import jax.numpy as np
from jax.scipy import stats
from jax.nn import log_sigmoid, log_softmax
from jax.scipy.special import logsumexp


def gen_logistic_log_pdf(x, mean, scale):
    """Element-wise log PDF of the logistic distribution

    Parameters
    ----------
    x : np.ndarray, (n_samples, n_features)
            data to be transformed

    mean : float,
            mean of the distribution

    scale : float,
            scale of the distribution

    Returns
    -------
    log_prob : np.ndarray, (n_samples, n_features)
            log probability of the distribution
    """

    # change of variables
    z = -(x - mean) / scale

    # log probability
    log_prob = stats.logistic.logpdf(z)

    return log_prob


def gen_logistic_log_cdf(x, mean, scale):
    """Element-wise log CDF of the logistic distribution

    Parameters
    ----------
    x : np.ndarray, (n_samples,)
            data to be transformed

    mean : float,
            mean of the distribution

    scale : float,
            scale of the distribution

    Returns
    -------
    log_prob : np.ndarray, (n_samples,)
            log cdf of the distribution
    """

    # change of variables
    z = -(x - mean) / scale

    # log cdf probability
    log_cdf_prob = log_sigmoid(z)

    return log_cdf_prob


def mixture_gaussian_log_pdf(x, prior_logits, means, scales):

    # normalize logit weights to 1
    prior_logits = log_softmax(prior_logits, axis=0)
    # print(prior_logits.shape)

    # calculate the log pdf
    log_pdfs = prior_logits + stats.norm.logpdf(x, means, scales)
    # print(log_pdfs.shape)

    # normalize distribution
    log_pdf = logsumexp(log_pdfs, axis=0)

    return log_pdf


def mixture_gaussian_log_cdf(x, prior_logits, means, scales):
    # normalize logit weights to 1
    prior_logits = jax.nn.log_softmax(prior_logits, axis=0)

    # calculate the log cdf
    log_cdfs = prior_logits + stats.norm.logcdf(x, means, scales)

    # normalize distribution
    log_cdf = jax.scipy.special.logsumexp(log_cdfs, axis=0)

    return log_cdf


def MixtureGaussCDF(n_components=5):
    """An implementation of a mixture Gaussian CDF layer.
    **Note**: the inverse function still has not been implemented yet!!

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """
    k = n_components

    def init_fun(rng, input_dim, **kwargs):
        n_components = kwargs.pop("n_components", k)
        # get random keys
        k1, k2 = random.split(rng, 2)

        # weight logits
        weight_logits = jax.random.normal(k1, (n_components, input_dim))
        # means
        means = jax.random.normal(k2, (n_components, input_dim))
        # scales
        scales = np.ones((n_components, input_dim))

        def direct_fun(params, inputs, **kwargs):
            weights, means, scales = params

            # z is the CDF of x
            # print(inputs.shape)
            log_z = jax.vmap(mixture_gaussian_log_cdf, in_axes=(0, None, None, None))(
                inputs, weights, means, scales
            )
            # print(log_z.shape)
            z = np.exp(log_z)

            # log_det = log_pdf(x)
            log_det = jax.vmap(mixture_gaussian_log_pdf, in_axes=(0, None, None, None))(
                inputs, weights, means, scales
            ).sum(axis=1)
            # print(z.shape, log_det.shape)

            return z, log_det

        def inverse_fun(params, inputs, **kwargs):
            weights, means, scales = params
            return [], []

        return (weight_logits, means, scales), direct_fun, inverse_fun

    return init_fun
