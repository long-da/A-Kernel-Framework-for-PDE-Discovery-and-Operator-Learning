import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


class RBF_kernel_f(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, X1, X2, ls):
        return jnp.exp(-1 / 2 * (((X1 - X2) / ls)**2).sum())


class poly_kernel_f(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, X1, X2, c, d):
        return ((X1 * X2).sum() + c)**d
