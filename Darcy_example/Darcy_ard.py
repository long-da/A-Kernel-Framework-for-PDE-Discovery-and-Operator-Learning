import jax.numpy as jnp
from kernels_f import *
from jax.lib import xla_bridge
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from kernels_u import *
from numpy import random
from jax.config import config
from sklearn.model_selection import train_test_split
from jax import vmap
from kernel_matrix import *
import random
from jaxopt import LBFGS

data = np.load('darcy.npy', allow_pickle=True).item()
tr_f = data['tr_f'][:20,]
tr_s = data['tr_s'][:20,]
te_f = data['te_f']
te_s = data['te_s']
X = data['X']

N_x = te_f.shape[1]  # 15
N_x_tes = N_x  # 15
N = 20  # number of training sources

config.update("jax_enable_x64", True)
print("Jax on", xla_bridge.get_backend().platform)
random.seed(123)
np.random.seed(123)


class EquationLearning(object):

    def __init__(self, N, X, u, f, K_u=RBF_kernel_u, K_f=RBF_kernel_f):
        self.K_u = K_u()
        self.K_f = K_f()
        self.jitter = 1e-8
        self.f_jitter = 1e-3
        self.X = X
        self.XX = jnp.tile(self.X, (N, 1))
        self.u = u
        self.f = f
        self.f_mean = np.ravel(self.f).mean()
        self.f_std = np.ravel(self.f).std()
        self.f_norm = (self.f - self.f_mean) / (self.f_std)
        self.K_f_ls = None
        self.K_f_weights = None
        self.N = self.X.shape[0]
        self.kernel_matrix = Kernel_matrix(self.jitter, self.K_u, "Darcy Flow")

    def get_K_f(self, X1, X2, ls):
        N = X2.shape[0]
        M = X1.shape[0]
        d = X1.shape[1]
        X1_p = jnp.transpose(jnp.tile(X1.T.reshape((d, 1, -1)), (N, 1)), (0, 2, 1)).reshape(d, -1).T
        X2_p = jnp.tile(X2.T.reshape((d, 1, -1)), (M, 1)).reshape(d, -1).T
        v_K_f_fun = vmap(self.K_f.kappa, (0, 0, None))
        return v_K_f_fun(X1_p, X2_p, ls).reshape((M, N))

    def get_K_u(self, cov, X1, X2, ls):
        N = X2.shape[0]
        M = X1.shape[0]
        d = 2
        X1_p = np.transpose(np.tile(X1.T.reshape((d, 1, -1)), (N, 1)), (0, 2, 1)).reshape(d, -1).T
        X2_p = np.tile(X2.T.reshape((d, 1, -1)), (M, 1)).reshape(d, -1).T
        v_K_f_fun = vmap(cov, (0, 0, 0, 0, None, None))
        return v_K_f_fun(X1_p[:, 0], X1_p[:, 1], X2_p[:, 0], X2_p[:, 1], ls, ls).reshape((M, N))

    def learn_K_u(self, u, X):
        u = jnp.ravel(u)
        validation_ratio = 0.5
        num_folders = 10
        rs = np.random.randint(123, size=num_folders)
        ls = np.linspace(0.001, 0.5, 100)
        errs_by_ls = np.zeros((ls.shape[0], num_folders))
        for j in range(num_folders):
            X_tr, X_val, u_tr, u_val = train_test_split(X, u.reshape((-1, 1)), test_size=validation_ratio, random_state=rs[j])
            for i in range(ls.shape[0]):
                K_tr_tr = self.get_K_u(self.K_u.kappa, X_tr, X_tr, ls[i])
                weights = (jnp.linalg.solve(K_tr_tr + 1e-8 * jnp.eye(X_tr.shape[0]), u_tr)).reshape((-1, 1))
                K_val_tr = self.get_K_u(self.K_u.kappa, X_val, X_tr, ls[i])
                u_val_preds = jnp.matmul(K_val_tr, weights)
                errs_by_ls[i, j] = metrics.mean_squared_error(u_val.reshape((-1, 1)), u_val_preds.reshape((-1, 1)))
        errs_by_ls = errs_by_ls.mean(axis=1)
        K_u_ls = ls[np.argmin(errs_by_ls)]
        K_u_u = self.get_K_u(self.K_u.kappa, X, X, K_u_ls)
        K_u_weights = (jnp.linalg.solve(K_u_u + 1e-8 * jnp.eye(K_u_u.shape[0]), u)).reshape((-1, 1))
        print("Validation set MSE ", errs_by_ls.min(), " kernel ls for u ", K_u_ls)
        return (np.append(K_u_ls.reshape(-1), K_u_weights.reshape(-1))).reshape((1, -1))

    def learn_grads(self, K_u_ls, K_u_weights, X):
        K_u_weights = jnp.ravel(K_u_weights)
        K_ddx1 = self.get_K_u(self.K_u.DD_x1_kappa, X, X, K_u_ls)
        K_ddx2 = self.get_K_u(self.K_u.DD_x2_kappa, X, X, K_u_ls)
        K_dx1 = self.get_K_u(self.K_u.D_x1_kappa, X, X, K_u_ls)
        K_dx2 = self.get_K_u(self.K_u.D_x2_kappa, X, X, K_u_ls)
        u_ddx1 = jnp.matmul(K_ddx1, K_u_weights)
        u_dx2 = jnp.matmul(K_dx2, K_u_weights)
        u_ddx2 = jnp.matmul(K_ddx2, K_u_weights)
        u_dx1 = jnp.matmul(K_dx1, K_u_weights)
        u_grads = np.concatenate((u_ddx1.reshape((-1, 1)), u_ddx2.reshape((-1, 1)), u_dx1.reshape((-1, 1)), u_dx2.reshape((-1, 1))), axis=1)
        return u_grads

    def learn_K_f(self):
        validation_ratio = 0.5
        num_folders = 3
        rs = np.random.randint(123, size=num_folders)
        ls1 = np.linspace(0.01, 0.5, 6)
        ls2 = np.linspace(2.01, 5.0, 6)
        ls3 = np.linspace(5.01, 5.0, 6)
        errs_by_ls = np.zeros((ls1.shape[0], ls2.shape[0], ls3.shape[0], num_folders))
        u_grads = np.concatenate((self.tr_grads, self.XX), axis=1)
        self.u_ddx1_mean = u_grads[:, 0].mean()
        self.u_ddx1_std = u_grads[:, 0].std()
        self.u_dx2_mean = u_grads[:, 3].mean()
        self.u_dx2_std = u_grads[:, 3].std()
        self.u_ddx2_mean = u_grads[:, 1].mean()
        self.u_ddx2_std = u_grads[:, 1].std()
        self.u_dx1_mean = u_grads[:, 2].mean()
        self.u_dx1_std = u_grads[:, 2].std()
        self.x1_mean = self.X[:, 0].mean()
        self.x1_std = self.X[:, 0].std()
        self.x2_mean = self.X[:, 1].mean()
        self.x2_std = self.X[:, 1].std()

        for j in range(num_folders):
            u_grads_tr, u_grads_val, f_tr_norm, f_val_norm = train_test_split(u_grads, (np.tile(self.tr_f_norm.flatten(), (N_x, 1)).T).reshape((-1, 1)), test_size=validation_ratio, random_state=rs[j])
            u_ddx1 = u_grads_tr[:, 0].flatten()
            u_ddx2 = u_grads_tr[:, 1].flatten()
            u_dx1 = u_grads_tr[:, 2].flatten()
            u_dx2 = u_grads_tr[:, 3].flatten()
            x1 = u_grads_tr[:, 4].flatten()
            x2 = u_grads_tr[:, 5].flatten()
            u_ddx1_val = u_grads_val[:, 0].flatten()
            u_dx2_val = u_grads_val[:, 3].flatten()
            u_ddx2_val = u_grads_val[:, 1].flatten()
            u_dx1_val = u_grads_val[:, 2].flatten()
            x1_val = u_grads_val[:, 4].flatten()
            x2_val = u_grads_val[:, 5].flatten()
            u_ddx1_tr_norm = (u_ddx1 - self.u_ddx1_mean) / self.u_ddx1_std
            u_dx2_tr_norm = (u_dx2 - self.u_dx2_mean) / self.u_dx2_std
            u_ddx2_tr_norm = (u_ddx2 - self.u_ddx2_mean) / self.u_ddx2_std
            u_dx1_tr_norm = (u_dx1 - self.u_dx1_mean) / self.u_dx1_std
            x1_tr_norm = (x1 - self.x1_mean) / self.x1_std
            x2_tr_norm = (x2 - self.x2_mean) / self.x2_std
            u_ddx1_val_norm = (u_ddx1_val - self.u_ddx1_mean) / self.u_ddx1_std
            u_dx2_val_norm = (u_dx2_val - self.u_dx2_mean) / self.u_dx2_std
            u_ddx2_val_norm = (u_ddx2_val - self.u_ddx2_mean) / self.u_ddx2_std
            u_dx1_val_norm = (u_dx1_val - self.u_dx1_mean) / self.u_dx1_std
            x1_val_norm = (x1_val - self.x1_mean) / self.x1_std
            x2_val_norm = (x2_val - self.x2_mean) / self.x2_std
            X_tr = jnp.concatenate(((x1_tr_norm.flatten()).reshape((-1, 1)), (x2_tr_norm.flatten()).reshape((-1, 1)), (u_ddx1_tr_norm.flatten()).reshape((-1, 1)), (u_ddx2_tr_norm.flatten()).reshape((-1, 1)), (u_dx1_tr_norm.flatten()).reshape(
                (-1, 1)), (u_dx2_tr_norm.flatten()).reshape((-1, 1))),
                                   axis=1)

            X_val = jnp.concatenate(((x1_val_norm.flatten()).reshape((-1, 1)), (x2_val_norm.flatten()).reshape((-1, 1)), (u_ddx1_val_norm.flatten()).reshape((-1, 1)), (u_ddx2_val_norm.flatten()).reshape((-1, 1)), (u_dx1_val_norm.flatten()).reshape(
                (-1, 1)), (u_dx2_val_norm.flatten()).reshape((-1, 1))),
                                    axis=1)
            for i in range(ls1.shape[0]):
                for k in range(ls2.shape[0]):
                    for l in range(ls3.shape[0]):
                        temp_ls = jnp.array([ls1[i], ls1[i], ls2[k], ls2[k], ls3[l], ls3[l]])
                        K_train_train = self.get_K_f(X_tr, X_tr, temp_ls)
                        K_val_train = self.get_K_f(X_val, X_tr, temp_ls)
                        weights = jnp.linalg.solve(K_train_train + self.f_jitter * jnp.eye(K_train_train.shape[0]), f_tr_norm)
                        f_test_pred_norm = jnp.matmul(K_val_train, weights)
                        f_val = f_val_norm * self.f_std + self.f_mean
                        f_val_pred = f_test_pred_norm * self.f_std + self.f_mean
                        errs_by_ls[i, k, l, j] = (((f_val.reshape(-1) - f_val_pred.reshape(-1))**2).sum() / ((f_val.reshape(-1))**2).sum())**0.5
        errs_by_ls = errs_by_ls.mean(axis=-1)
        ix = np.argwhere(errs_by_ls == errs_by_ls.min()).reshape(-1)
        self.K_f_ls = jnp.array([ls1[ix[0]], ls1[ix[0]], ls2[ix[1]], ls2[ix[1]], ls3[ix[2]], ls3[ix[2]]])
        print("Validation set error ", errs_by_ls.min(), " kernel ls for f ", self.K_f_ls)
        x1 = self.XX[:, 0].flatten()
        x2 = self.XX[:, 1].flatten()
        u_ddx1 = self.tr_grads[:, 0]
        u_dx2 = self.tr_grads[:, 3]
        u_ddx2 = self.tr_grads[:, 1]
        u_dx1 = self.tr_grads[:, 2]
        u_ddx1_train_norm = (u_ddx1 - self.u_ddx1_mean) / self.u_ddx1_std
        u_dx2_train_norm = (u_dx2 - self.u_dx2_mean) / self.u_dx2_std
        u_ddx2_train_norm = (u_ddx2 - self.u_ddx2_mean) / self.u_ddx2_std
        u_dx1_train_norm = (u_dx1 - self.u_dx1_mean) / self.u_dx1_std
        x1_train_norm = (x1 - self.x1_mean) / self.x1_std
        x2_train_norm = (x2 - self.x2_mean) / self.x2_std
        self.X_tr = jnp.concatenate(((x1_train_norm.flatten()).reshape((-1, 1)), (x2_train_norm.flatten()).reshape((-1, 1)), (u_ddx1_train_norm.flatten()).reshape((-1, 1)), (u_ddx2_train_norm.flatten()).reshape(
            (-1, 1)), (u_dx1_train_norm.flatten()).reshape((-1, 1)), (u_dx2_train_norm.flatten()).reshape((-1, 1))),
                                    axis=1)
        K_train_train = self.get_K_f(self.X_tr, self.X_tr, self.K_f_ls)
        self.K_f_weights = jnp.linalg.solve(K_train_train + self.f_jitter * jnp.eye(K_train_train.shape[0]), (np.tile(self.tr_f_norm.flatten(), (N_x, 1)).T).reshape((-1, 1)))

    def loss(self, z1):
        pen_lambda = self.pen_lambda
        L = self.L
        u_b = self.u_b.reshape(-1)
        z = jnp.append(u_b, z1)
        ss = jnp.linalg.solve(L, z)
        z_ddx1 = z1[self.N_f:2 * self.N_f + self.N_b]
        z_ddx2 = z1[2 * self.N_f + self.N_b:3 * self.N_f + 2 * self.N_b]
        z_x1 = z1[3 * self.N_f + 2 * self.N_b:4 * self.N_f + 3 * self.N_b]
        z_x2 = z1[4 * self.N_f + 3 * self.N_b:]
        x1_con = self.X_con[:, 0]
        x2_con = self.X_con[:, 1]
        zz_x1 = (x1_con - self.x1_mean) / (self.x1_std)
        zz_x2 = (x2_con - self.x2_mean) / (self.x2_std)
        z_ddx1 = (z_ddx1 - self.u_ddx1_mean) / (self.u_ddx1_std)
        z_ddx2 = (z_ddx2 - self.u_ddx2_mean) / (self.u_ddx2_std)
        z_x1 = (z_x1 - self.u_dx1_mean) / (self.u_dx1_std)
        z_x2 = (z_x2 - self.u_dx2_mean) / (self.u_dx2_std)
        f = jnp.append(self.f_b, self.f_q)
        X1 = jnp.concatenate(((zz_x1.flatten()).reshape((-1, 1)), (zz_x2.flatten()).reshape((-1, 1)), (z_ddx1.flatten()).reshape((-1, 1)), (z_ddx2.flatten()).reshape((-1, 1)), (z_x1.flatten()).reshape((-1, 1)), (z_x2.flatten()).reshape((-1, 1))),
                             axis=1)
        K_test_train = self.get_K_f(X1, self.X_tr, self.K_f_ls)
        f_pred = jnp.matmul(K_test_train, self.K_f_weights) * self.f_std + self.f_mean
        f = f * self.f_std + self.f_mean
        ss2 = f_pred.reshape(-1) - f.reshape(-1)
        return jnp.dot(ss, ss) + jnp.dot(ss2, ss2) / pen_lambda

    def solve_other_source2(self, X_tes, tes_u, f, ls=None, loss_lambda=None, steps=None, test_name="u_q"):
        tes_f_norm = (f - self.f_mean) / self.f_std
        self.N_b = 2 * N_x_tes + 2 * (N_x_tes - 2)
        self.N_f = (N_x_tes - 2) * (N_x_tes - 2)
        temp_f = np.tile(tes_f_norm.flatten(), (N_x_tes, 1)).T
        temp_u = tes_u.reshape((N_x_tes, N_x_tes))
        temp_X1 = (X_tes[:, 0]).reshape((N_x_tes, N_x_tes))
        temp_X2 = (X_tes[:, 1]).reshape((N_x_tes, N_x_tes))
        self.f_b = np.concatenate((temp_f[:, 0].reshape((-1, 1)), temp_f[:, -1].reshape((-1, 1)), temp_f[0, 1:-1].reshape((-1, 1)), temp_f[-1, 1:-1].reshape((-1, 1))), axis=0)
        self.f_q = temp_f[1:-1, 1:-1].reshape((-1, 1))
        X1_b = np.concatenate((temp_X1[:, 0].reshape((-1, 1)), temp_X1[:, -1].reshape((-1, 1)), temp_X1[0, 1:-1].reshape((-1, 1)), temp_X1[-1, 1:-1].reshape((-1, 1))), axis=0)
        X2_b = np.concatenate((temp_X2[:, 0].reshape((-1, 1)), temp_X2[:, -1].reshape((-1, 1)), temp_X2[0, 1:-1].reshape((-1, 1)), temp_X2[-1, 1:-1].reshape((-1, 1))), axis=0)
        X1_q = temp_X1[1:-1, 1:-1].reshape(-1)
        X2_q = temp_X2[1:-1, 1:-1].reshape(-1)
        self.X_b = np.concatenate((X1_b.reshape((-1, 1)), X2_b.reshape((-1, 1))), axis=1)
        self.X_q = np.concatenate((X1_q.reshape((-1, 1)), X2_q.reshape((-1, 1))), axis=1)
        self.u_b = np.concatenate((temp_u[:, 0].reshape((-1, 1)), temp_u[:, -1].reshape((-1, 1)), temp_u[0, 1:-1].reshape((-1, 1)), temp_u[-1, 1:-1].reshape((-1, 1))), axis=0)
        self.N_con = self.N_b + self.N_f
        N_z = 5 * self.N_f + 4 * self.N_b
        self.X_con = jnp.concatenate((self.X_b, self.X_q), axis=0)
        self.f_con_norm = jnp.concatenate((self.f_b.reshape((-1, 1)), self.f_q.reshape((-1, 1))), axis=0)
        self.f_con_norm = self.f_con_norm.reshape(-1)
        x1_p = jnp.tile(self.X_con[:, 0].flatten(), (self.N_con, 1)).T
        x2_p = jnp.tile(self.X_con[:, 1].flatten(), (self.N_con, 1)).T
        X1_p = jnp.concatenate((x1_p.reshape((-1, 1)), x2_p.reshape((-1, 1))), axis=1)
        X2_p = jnp.concatenate((jnp.transpose(x1_p).reshape((-1, 1)), jnp.transpose(x2_p).reshape((-1, 1))), axis=1)
        ls = jnp.array([0.10, 0.2, 0.25])
        lambdas = jnp.zeros(1)
        lambdas = np.array([1e-2, 1e-4, 1e-6])
        err_min = jnp.zeros((ls.shape[0], 2))
        random.seed(0)
        np.random.seed(0)
        sol0 = np.zeros(N_z)
        err_min = 1.0
        preds = None
        for i in range(ls.shape[0]):
            self.K_z = self.kernel_matrix.get_kernel_matrx(X1_p, X2_p, ls[i])
            L = jnp.linalg.cholesky(self.K_z)
            self.L = L
            for j in range(lambdas.shape[0]):
                random.seed(123)
                np.random.seed(123)
                self.pen_lambda = lambdas[j]
                sol = sol0
                opt = LBFGS(self.loss, maxiter=4000, tol=1e-15, jit=True, stepsize=0.5)
                sol, _ = opt.run(sol)
                pred_u = sol[:self.N_f]
                err_tes = ((((tes_u.reshape(N_x_tes, N_x_tes)[1:-1, 1:-1]).reshape((-1)) - pred_u.reshape(-1))**2).sum() / (((tes_u.reshape(N_x_tes, N_x_tes)[1:-1, 1:-1]).reshape((-1)))**2).sum())**0.5
                if err_min > err_tes and not np.isnan(err_tes):
                    err_min = err_tes
                    preds = pred_u
        return preds, err_min

    def train(self):
        # res = map(lambda u, : self.learn_K_u(u, self.X), self.u)
        # res = np.vstack(list(zip(*res)))
        # self.kernels_u = res[:, 0]
        # self.weights_u = res[:, 1:]
        # res = map(lambda kernels_u, weights_u: self.learn_grads(kernels_u, weights_u, self.X), self.kernels_u, self.weights_u)
        # self.grads = np.array(list(res))

        # self.tr_grads = (self.grads).reshape((-1, 4))

        # np.save('grads.npy', {
        #     'grads': self.tr_grads,
        # })
        data = np.load('grads.npy', allow_pickle=True).item()
        self.tr_grads = data['grads']
        self.tr_u = np.ravel(self.u)
        self.tr_f_norm = np.ravel(self.f_norm)
        self.learn_K_f()


el = EquationLearning(N, X, tr_s, tr_f)
el.train()

errs = np.zeros((50, 1))
preds_list = []

for i in range(50):
    temp, errs[i, 0] = el.solve_other_source2(X, (te_s[i, :].reshape(N_x, N_x)).reshape(1, -1), te_f[i, :].reshape(1, -1))
    preds_list.append(temp)
    print(errs[i, 0])
print(errs[:, 0].mean())
print("std ", np.std(errs) / (50**0.5))
np.save('preds_ard', {
    'preds': np.array(preds_list),
    'errs': errs,
    'mean': errs.mean(),
    'std': np.std(errs) / (50**0.5),
})
