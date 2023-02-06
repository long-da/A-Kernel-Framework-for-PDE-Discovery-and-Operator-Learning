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
        if self.PDE == "Darcy Flow":
            K_z = jnp.zeros((5 * N, 5 * N))
            K_u_u = vmap(self.K_u.kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx1_ddx1 = vmap(self.K_u.DD_x1_DD_y1_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx1 = vmap(self.K_u.DD_x1_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx2_ddx2 = vmap(self.K_u.DD_x2_DD_y2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx2 = vmap(self.K_u.DD_x2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_dx1_dx1 = vmap(self.K_u.D_x1_D_y1_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_dx1 = vmap(self.K_u.D_x1_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_dx2_dx2 = vmap(self.K_u.D_x2_D_y2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_dx2 = vmap(self.K_u.D_x2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx1_ddx2 = vmap(self.K_u.DD_x1_DD_y2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx1_dx1 = vmap(self.K_u.DD_x1_D_y1_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx2_dx1 = vmap(self.K_u.DD_x2_D_y1_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx2_dx2 = vmap(self.K_u.DD_x2_D_y2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_ddx1_dx2 = vmap(self.K_u.DD_x1_D_y2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_dx1_dx2 = vmap(self.K_u.D_x1_D_y2_kappa, (0, 0, 0, 0, None, None))(X1[:, 0], X1[:, 1], X2[:, 0], X2[:, 1], ls, ls).reshape(N, N)
            K_z = K_z.at[:N, :N].set(K_u_u)
            K_z = K_z.at[N:2 * N, N:2 * N].set(K_ddx1_ddx1)
            K_z = K_z.at[2 * N:3 * N, 2 * N:3 * N].set(K_ddx2_ddx2)
            K_z = K_z.at[3 * N:4 * N, 3 * N:4 * N].set(K_dx1_dx1)
            K_z = K_z.at[4 * N:5 * N, 4 * N:5 * N].set(K_dx2_dx2)
            K_z = K_z.at[N:2 * N, :N].set(K_ddx1)
            K_z = K_z.at[:N, N:2 * N].set(K_ddx1.T)
            K_z = K_z.at[2 * N:3 * N, :N].set(K_ddx2)
            K_z = K_z.at[:N, 2 * N:3 * N].set(K_ddx2.T)
            K_z = K_z.at[3 * N:4 * N, :N].set(K_dx1)
            K_z = K_z.at[:N, 3 * N:4 * N].set(K_dx1.T)
            K_z = K_z.at[4 * N:5 * N, :N].set(K_dx2)
            K_z = K_z.at[:N, 4 * N:5 * N].set(K_dx2.T)
            K_z = K_z.at[N:2 * N, 2 * N:3 * N].set(K_ddx1_ddx2)
            K_z = K_z.at[2 * N:3 * N, N:2 * N].set(K_ddx1_ddx2.T)
            K_z = K_z.at[N:2 * N, 3 * N:4 * N].set(K_ddx1_dx1)
            K_z = K_z.at[3 * N:4 * N, N:2 * N].set(K_ddx1_dx1.T)
            K_z = K_z.at[N:2 * N, 4 * N:5 * N].set(K_ddx1_dx2)
            K_z = K_z.at[4 * N:5 * N, N:2 * N].set(K_ddx1_dx2.T)
            K_z = K_z.at[2 * N:3 * N, 3 * N:4 * N].set(K_ddx2_dx1)
            K_z = K_z.at[3 * N:4 * N, 2 * N:3 * N].set(K_ddx2_dx1.T)
            K_z = K_z.at[2 * N:3 * N, 4 * N:5 * N].set(K_ddx2_dx2)
            K_z = K_z.at[4 * N:5 * N, 2 * N:3 * N].set(K_ddx2_dx2.T)
            K_z = K_z.at[3 * N:4 * N, 4 * N:5 * N].set(K_dx1_dx2)
            K_z = K_z.at[4 * N:5 * N, 3 * N:4 * N].set(K_dx1_dx2.T)
            K_z = K_z + self.jitter * jnp.eye(5 * N)
            return K_z
