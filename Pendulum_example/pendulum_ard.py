import jax.numpy as jnp
from kernels_f import *
from kernels_u import *
from jax.lib import xla_bridge
import numpy as np
from jax.config import config
from sklearn.model_selection import train_test_split
from jax import vmap
from kernel_matrix import *
import random
from jax import jacfwd

N_p = 20
data = np.load('pendulum.npy', allow_pickle=True).item()
tr_f = data['tr_f'][:N_p, :]
tr_s = data['tr_s'][:N_p, :]
te_f = data['te_f']
te_s = data['te_s']
X0 = data['X0']
te_f0 = data['te_f0']
te_s0 = data['te_s0']
X = data['X']
N_x = te_f.shape[1]
N_x_tes = N_x
tr_s1 = tr_s[:, :N_x]
tr_s2 = tr_s[:, N_x:]
te_s1 = te_s[:, :N_x]
te_s2 = te_s[:, N_x:]

config.update("jax_enable_x64", True)
print("Jax on", xla_bridge.get_backend().platform)
random.seed(123)
np.random.seed(123)


class EquationLearning(object):

    def __init__(self, X, u1, u2, f, K_u=RBF_kernel_u_1d, K_f=RBF_kernel_f):
        self.K_u = K_u()
        self.K_f = K_f()
        self.jitter = 1e-8
        self.f_jitter = 1e-5
        self.u1 = u1
        self.u2 = u2
        self.f = f
        self.f_mean = np.ravel(self.f).mean()
        self.f_std = np.ravel(self.f).std()
        self.f_norm = (self.f - self.f_mean) / (self.f_std)
        self.K_f_ls = None
        self.K_f_weights = None
        self.kernel_matrix = Kernel_matrix(self.jitter, self.K_u, "Pendulum")

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
        d = 1
        X1_p = np.transpose(np.tile(X1.T.reshape((d, 1, -1)), (N, 1)), (0, 2, 1)).reshape(d, -1).T
        X2_p = np.tile(X2.T.reshape((d, 1, -1)), (M, 1)).reshape(d, -1).T
        v_K_f_fun = vmap(cov, (0, 0, None))
        return v_K_f_fun(X1_p[:, 0], X2_p[:, 0], ls).reshape((M, N))

    def learn_K_u(self, u, X):
        u = jnp.ravel(u)
        validation_ratio = 0.5
        num_folders = 5
        rs = np.random.randint(123, size=num_folders)
        ls = np.linspace(0.001, 1.0, 100)
        errs_by_ls = np.zeros((ls.shape[0], num_folders))
        for j in range(num_folders):
            X_tr, X_val, u_tr, u_val = train_test_split(X, u.reshape((-1, 1)), test_size=validation_ratio, random_state=rs[j])
            for i in range(ls.shape[0]):
                K_tr_tr = self.get_K_u(self.K_u.kappa, X_tr, X_tr, ls[i])
                weights = (jnp.linalg.solve(K_tr_tr + 1e-8 * jnp.eye(X_tr.shape[0]), u_tr)).reshape((-1, 1))
                K_val_tr = self.get_K_u(self.K_u.kappa, X_val, X_tr, ls[i])
                u_val_preds = jnp.matmul(K_val_tr, weights)
                errs_by_ls[i, j] = (((u_val.reshape((-1, 1)) - u_val_preds.reshape((-1, 1)))**2).sum() / (u_val.reshape((-1, 1))**2).sum())**0.5
        errs_by_ls = errs_by_ls.mean(axis=1)
        K_u_ls = ls[np.argmin(errs_by_ls)]
        K_u_u = self.get_K_u(self.K_u.kappa, X, X, K_u_ls)
        K_u_weights = (jnp.linalg.solve(K_u_u + 1e-8 * jnp.eye(K_u_u.shape[0]), u)).reshape((-1, 1))
        print("Validation set MSE ", errs_by_ls.min(), " kernel ls for u ", K_u_ls)
        return (np.append(K_u_ls.reshape(-1), K_u_weights.reshape(-1))).reshape((1, -1))

    def learn_grads(self, K_u_ls, K_u_weights, X):
        K_u_weights = jnp.ravel(K_u_weights)
        K_dx1 = self.get_K_u(self.K_u.D_x1_kappa, X, X, K_u_ls)
        u_dx1 = jnp.matmul(K_dx1, K_u_weights)
        u_grads = u_dx1
        return u_grads

    def learn_K_f1(self):
        validation_ratio = 0.5
        num_folders = 10
        rs = np.random.randint(123, size=num_folders)
        ls = np.linspace(0.01, 1.0, 100)
        errs_by_ls = np.zeros((ls.shape[0], num_folders))
        u_grads = self.tr_grads1
        self.u1_dx1_mean = u_grads[:, 0].mean()
        self.u1_dx1_std = u_grads[:, 0].std()
        for j in range(num_folders):
            u_grads_tr, u_grads_val, u2_tr_norm, u2_val_norm = train_test_split(u_grads, self.tr_u2_norm.reshape((-1, 1)), test_size=validation_ratio, random_state=rs[j])
            u1_dx1 = u_grads_tr[:, 0].flatten()
            u1_dx1_val = u_grads_val[:, 0].flatten()
            u1_dx1_tr_norm = (u1_dx1 - self.u1_dx1_mean) / self.u1_dx1_std
            u1_dx1_val_norm = (u1_dx1_val - self.u1_dx1_mean) / self.u1_dx1_std
            X_tr = u1_dx1_tr_norm.reshape((-1, 1))
            X_val = u1_dx1_val_norm.reshape((-1, 1))
            for i in range(ls.shape[0]):
                K_train_train = self.get_K_f(X_tr, X_tr, ls[i])
                K_val_train = self.get_K_f(X_val, X_tr, ls[i])
                weights = jnp.linalg.solve(K_train_train + 1e-5 * jnp.eye(K_train_train.shape[0]), u2_tr_norm)
                f_test_pred_norm = jnp.matmul(K_val_train, weights)
                f_val = u2_val_norm * self.tr_u2_std + self.tr_u2_mean
                f_val_pred = f_test_pred_norm * self.tr_u2_std + self.tr_u2_mean
                errs_by_ls[i, j] = (((f_val.reshape(-1) - f_val_pred.reshape(-1))**2).sum() / (f_val.reshape(-1)**2).sum())**0.5
        errs_by_ls = errs_by_ls.mean(axis=1)
        self.K_f_ls1 = ls[np.argmin(errs_by_ls)]
        print("Validation set error ", errs_by_ls.min(), " kernel ls for f ", self.K_f_ls1)
        u_dx1 = self.tr_grads1[:, 0]
        u_ddx1_train_norm = (u_dx1 - self.u1_dx1_mean) / self.u1_dx1_std
        self.X_tr1 = u_ddx1_train_norm.reshape((-1, 1))
        K_train_train = self.get_K_f(self.X_tr1, self.X_tr1, self.K_f_ls1)
        self.K_f_weights1 = jnp.linalg.solve(K_train_train + 1e-5 * jnp.eye(K_train_train.shape[0]), self.tr_u2_norm.reshape((-1, 1)))

    def learn_K_f2(self):
        validation_ratio = 0.5
        num_folders = 10
        rs = np.random.randint(123, size=num_folders)
        ls = np.linspace(0.01, 3.0, 100)
        errs_by_ls = np.zeros((ls.shape[0], num_folders))
        u_grads = np.concatenate((self.tr_grads2, self.tr_u1), axis=1)
        self.u2_dx1_mean = u_grads[:, 0].mean()
        self.u2_dx1_std = u_grads[:, 0].std()
        self.u1_mean = u_grads[:, 1].mean()
        self.u1_std = u_grads[:, 1].std()

        for j in range(num_folders):
            u_grads_tr, u_grads_val, f_tr_norm, f_val_norm = train_test_split(u_grads, self.tr_f_norm.reshape((-1, 1)), test_size=validation_ratio, random_state=rs[j])
            u2_dx1 = u_grads_tr[:, 0].flatten()
            u2_dx1_val = u_grads_val[:, 0].flatten()
            u2_dx1_tr_norm = (u2_dx1 - self.u2_dx1_mean) / self.u2_dx1_std
            u2_dx1_val_norm = (u2_dx1_val - self.u2_dx1_mean) / self.u2_dx1_std
            u1_tr = u_grads_tr[:, 1].flatten()
            u1_val = u_grads_val[:, 1].flatten()
            u1_tr_norm = (u1_tr - self.u1_mean) / self.u1_std
            u1_val_norm = (u1_val - self.u1_mean) / self.u1_std
            X_tr = jnp.concatenate(((u2_dx1_tr_norm.flatten()).reshape((-1, 1)), (u1_tr_norm.flatten()).reshape((-1, 1))), axis=1)
            X_val = jnp.concatenate(((u2_dx1_val_norm.flatten()).reshape((-1, 1)), (u1_val_norm.flatten()).reshape((-1, 1))), axis=1)
            for i in range(ls.shape[0]):
                K_train_train = self.get_K_f(X_tr, X_tr, ls[i])
                K_val_train = self.get_K_f(X_val, X_tr, ls[i])
                weights = jnp.linalg.solve(K_train_train + self.f_jitter * jnp.eye(K_train_train.shape[0]), f_tr_norm)
                f_test_pred_norm = jnp.matmul(K_val_train, weights)
                f_val = f_val_norm * self.f_std + self.f_mean
                f_val_pred = f_test_pred_norm * self.f_std + self.f_mean
                errs_by_ls[i, j] = (((f_val.reshape(-1) - f_val_pred.reshape(-1))**2).sum() / (f_val.reshape(-1)**2).sum())**0.5
        errs_by_ls = errs_by_ls.mean(axis=1)
        self.K_f_ls2 = ls[np.argmin(errs_by_ls)]
        print("Validation set error ", errs_by_ls.min(), " kernel ls for f ", self.K_f_ls2)
        u2_dx1 = self.tr_grads2[:, 0]
        u1_tr = self.tr_u1
        u2_dx1_train_norm = (u2_dx1 - self.u2_dx1_mean) / self.u2_dx1_std
        u1_train_norm = (u1_tr - self.u1_mean) / self.u1_std
        self.X_tr2 = jnp.concatenate(((u2_dx1_train_norm.flatten()).reshape((-1, 1)), (u1_train_norm.flatten()).reshape((-1, 1))), axis=1)
        K_train_train = self.get_K_f(self.X_tr2, self.X_tr2, self.K_f_ls2)
        self.K_f_weights2 = jnp.linalg.solve(K_train_train + self.f_jitter * jnp.eye(K_train_train.shape[0]), self.tr_f_norm.reshape((-1, 1)))

    def get_r(self, z1):
        pen_lambda = self.pen_lambda
        L1 = self.L1
        L2 = self.L2
        u_b1 = self.u_b1.reshape(-1)
        u_b2 = self.u_b2.reshape(-1)
        z11 = jnp.append(u_b1, z1[:self.N_z])
        z22 = jnp.append(u_b2, z1[self.N_z:])
        ss11 = jnp.linalg.solve(L1, z11)
        ss22 = jnp.linalg.solve(L2, z22)
        z_dx1_1 = z1[self.N_f:self.N_z]
        z_u_2 = jnp.append(u_b2, z1[self.N_z:self.N_z + self.N_f])
        z_dx1_1_norm = (z_dx1_1 - self.u1_dx1_mean) / self.u1_dx1_std
        z_u_1 = jnp.append(u_b1, z1[:self.N_f])
        z_u_1_norm = (z_u_1 - self.u1_mean) / self.u1_std
        X1 = z_dx1_1_norm.reshape((-1, 1))
        K_test_train = self.get_K_f(X1, self.X_tr1, self.K_f_ls1)
        z_u_2_pred = jnp.matmul(K_test_train, self.K_f_weights1) * self.tr_u2_std + self.tr_u2_mean
        f = self.f_con_norm * self.f_std + self.f_mean
        z_dx1_2 = z1[self.N_z + self.N_f:]
        z_dx1_2_norm = (z_dx1_2 - self.u2_dx1_mean) / self.u2_dx1_std
        X1 = jnp.concatenate(((z_dx1_2_norm.flatten()).reshape((-1, 1)), (z_u_1_norm.flatten()).reshape((-1, 1))), axis=1)
        K_test_train = self.get_K_f(X1, self.X_tr2, self.K_f_ls2)
        f_pred = jnp.matmul(K_test_train, self.K_f_weights2) * self.f_std + self.f_mean
        ss33 = z_u_2_pred.reshape(-1) - z_u_2.reshape(-1)
        ss44 = f_pred.reshape(-1) - f.reshape(-1)
        out = jnp.append(ss11, ss22)
        out = jnp.append(out, ss33 / pen_lambda**0.5)
        out = jnp.append(out, ss44 / pen_lambda**0.5)
        return out

    def solve_other_source(self, X_tes, tes_u, f):
        tes_f_norm = ((f - self.f_mean) / self.f_std).flatten()
        X_tes = X_tes.flatten()
        tes_u = tes_u.flatten()
        self.N_b = 1
        self.N_f = N_x_tes - 1
        self.f_b = tes_f_norm[0]
        self.f_q = tes_f_norm[1:]
        X_b = X_tes[0]
        X_q = X_tes[1:]
        self.X_b = X_b
        self.X_q = X_q
        u1 = tes_u[:N_x_tes]
        u2 = tes_u[N_x_tes:]
        self.u_b1 = u1[0]
        self.u_b2 = u2[0]
        self.N_con = self.N_b + self.N_f
        N_z = 2 * self.N_f + self.N_b
        self.N_z = N_z
        self.X_con = jnp.concatenate((self.X_b.reshape((-1, 1)), self.X_q.reshape((-1, 1))), axis=0)
        self.f_con_norm = jnp.concatenate((self.f_b.reshape((-1, 1)), self.f_q.reshape((-1, 1))), axis=0)
        self.f_con_norm = self.f_con_norm.reshape(-1)
        x1_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x1_p.flatten()
        X2_p = jnp.transpose(x1_p).flatten()
        ls1 = jnp.array([0.28])
        ls2 = jnp.array([0.28])
        lambdas = np.array([1e-12, 1e-6, 1e-8])
        random.seed(123)
        np.random.seed(123)
        sol0 = np.zeros(2 * N_z)
        err_min = 1.0
        best_preds = None
        for i in range(ls1.shape[0]):
            for k in range(ls2.shape[0]):
                self.K_z1, self.K_z2 = self.kernel_matrix.get_kernel_matrx(X1_p, X2_p, ls1[i], ls2[k])
                L1 = jnp.linalg.cholesky(self.K_z1)
                self.L1 = L1
                L2 = jnp.linalg.cholesky(self.K_z2)
                self.L2 = L2
                for j in range(lambdas.shape[0]):
                    random.seed(123)
                    np.random.seed(123)
                    self.pen_lambda = lambdas[j]
                    sol = sol0
                    for _ in range(1, 40):
                        r = self.get_r(sol).reshape((-1, 1))
                        jac = jacfwd(self.get_r)(sol)
                        temp = jnp.linalg.solve(jnp.matmul(jac.T, jac), jnp.matmul(jac.T, r).reshape(-1))
                        sol = sol - 1.0 * temp
                    pred_u2 = sol[N_z:N_z + self.N_f]
                    err = ((((u2[1:]).reshape((-1)) - pred_u2.reshape(-1))**2).sum() / (u2[1:]**2).sum())**0.5
                    if err_min > err:
                        temp_u = jnp.append(self.u_b2.reshape(-1), pred_u2)
                        K_u_u = self.get_K_u(self.K_u.kappa, self.X_con, self.X_con, ls2[k]) + self.jitter * jnp.eye((self.X_con.reshape(-1)).shape[0])
                        weights = jnp.linalg.solve(K_u_u, temp_u)
                        K_tes_u = self.get_K_u(self.K_u.kappa, X0, self.X_con, ls2[k])
                        pred_tes = jnp.matmul(K_tes_u, weights)
                        err_min = err
                        best_preds = pred_tes
        return best_preds, err_min

    def train(self):
        # print("Grad 1 ")
        # res1 = map(lambda u, : self.learn_K_u(u, self.X), self.u1)
        # res1 = np.vstack(list(zip(*res1)))
        # print("Grad 2 ")
        # res2 = map(lambda u, : self.learn_K_u(u, self.X), self.u2)
        # res2 = np.vstack(list(zip(*res2)))
        # self.kernels_u1 = res1[:, 0]
        # self.weights_u1 = res1[:, 1:]
        # self.kernels_u2 = res2[:, 0]
        # self.weights_u2 = res2[:, 1:]
        # res1 = map(lambda kernels_u, weights_u: self.learn_grads(kernels_u, weights_u, self.X), self.kernels_u1, self.weights_u1)
        # self.grads1 = np.array(list(res1))
        # res2 = map(lambda kernels_u, weights_u: self.learn_grads(kernels_u, weights_u, self.X), self.kernels_u2, self.weights_u2)
        # self.grads2 = np.array(list(res2))
        self.tr_u1 = np.ravel(self.u1).reshape((-1, 1))
        self.tr_u2 = np.ravel(self.u2).reshape((-1, 1))
        self.tr_u2_mean = self.tr_u2.mean()
        self.tr_u2_std = self.tr_u2.std()
        self.tr_u2_norm = (self.tr_u2 - self.tr_u2_mean) / self.tr_u2_std
        # self.tr_grads1 = (self.grads1).reshape((-1, 1))
        # self.tr_grads2 = (self.grads2).reshape((-1, 1))

        # np.save('grads.npy', {
        #     'grads1': self.tr_grads1,
        #     'grads2': self.tr_grads2,
        # })
        data = np.load('grads.npy', allow_pickle=True).item()
        self.tr_grads1 = data['grads1']
        self.tr_grads2 = data['grads2']
        self.tr_f_norm = np.ravel(self.f_norm).reshape((-1, 1))
        self.learn_K_f1()
        self.learn_K_f2()


el = EquationLearning(X, tr_s1, tr_s2, tr_f)
el.train()

errs = np.zeros((50, 1))
preds_list = []
for i in range(50):
    print("test case ", i)
    temp, errs[i, 0] = el.solve_other_source(X, te_s[i, :], te_f[i, :])
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
