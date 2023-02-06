import jax.numpy as jnp
from kernels_f import *
from kernels_u import *
from jax import vmap


class Kernel_matrix(object):

    def __init__(self, jitter, K_u, PDE):
        self.PDE = PDE
        self.jitter = jitter
        self.K_u = K_u

    @partial(jit, static_argnums=(0,))
    def get_kernel_matrx(self, X1, X2, ls, ls2=None):
        N = int((X1.shape[0])**0.5)
        if self.PDE == "Pendulum":
            ls1 = ls
            K_z1 = jnp.zeros((2 * N, 2 * N))
            K_z2 = jnp.zeros((2 * N, 2 * N))
            K_u_u_1 = vmap(self.K_u.kappa, (0, 0, None))(X1.flatten(), X2.flatten(), ls1).reshape(N, N)
            K_u_u_2 = vmap(self.K_u.kappa, (0, 0, None))(X1.flatten(), X2.flatten(), ls2).reshape(N, N)
            K_dx1_1 = vmap(self.K_u.D_x1_kappa, (0, 0, None))(X1.flatten(), X2.flatten(), ls1).reshape(N, N)
            K_dx1_2 = vmap(self.K_u.D_x1_kappa, (0, 0, None))(X1.flatten(), X2.flatten(), ls2).reshape(N, N)
            K_dx1_dx1_1 = vmap(self.K_u.D_x1_D_y1_kappa, (0, 0, None))(X1.flatten(), X2.flatten(), ls1).reshape(N, N)
            K_dx1_dx1_2 = vmap(self.K_u.D_x1_D_y1_kappa, (0, 0, None))(X1.flatten(), X2.flatten(), ls2).reshape(N, N)
            K_z1 = K_z1.at[:N, :N].set(K_u_u_1)
            K_z2 = K_z2.at[:N, :N].set(K_u_u_2)
            K_z1 = K_z1.at[N:2 * N, N:2 * N].set(K_dx1_dx1_1)
            K_z2 = K_z2.at[N:2 * N, N:2 * N].set(K_dx1_dx1_2)
            K_z1 = K_z1.at[N:2 * N, :N].set(K_dx1_1)
            K_z2 = K_z2.at[N:2 * N, :N].set(K_dx1_2)
            K_z1 = K_z1.at[:N, N:2 * N].set(K_dx1_1.T)
            K_z2 = K_z2.at[:N, N:2 * N].set(K_dx1_2.T)
            K_z1 = K_z1 + self.jitter * jnp.eye(2 * N)
            K_z2 = K_z2 + self.jitter * jnp.eye(2 * N)
            return K_z1, K_z2
