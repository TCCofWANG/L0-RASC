import time

from scipy.linalg import toeplitz
import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class L0RANMFNN_Layer(nn.Module):
    def __init__(self,X,para):

        super(L0RANMFNN_Layer, self).__init__()
        self.X = X
        self.rows, self.cols = X[0].shape
        self.rank = para['R']
        self.Block_num = para['Block_num']

        self.params = {}

        for idx in range(self.Block_num):
            self.params[f'lambda_tr{idx}'] = nn.Parameter(torch.full((self.rows, self.rank), para['lambda_tr'], dtype=torch.float32))
            self.params[f'K1{idx}'] = nn.Parameter(torch.full((self.rows, self.rank), 1, dtype=torch.float32))
            self.params[f'lambda_pt{idx}'] = nn.Parameter(torch.full((self.rank, self.cols), para['lambda_pt'], dtype=torch.float32))
            self.params[f'K2{idx}'] = nn.Parameter(torch.full((self.rank, self.cols), 1, dtype=torch.float32))


        self.lambda_a = torch.tensor(para['lambda_a'], dtype=torch.float32)
        self.alpha = torch.tensor(para['alpha'], dtype=torch.float32)
        self.max_iter = para['max_iter']
        self.epsilon = para['epsilon']
        self.tol = para['tol']
        self.t0 = para['t0']
        self.train_data_num = para['training set size']
        self.test_data_num = para['testing set size']
        torch.manual_seed(1)
        self.TTt = self.temporal_difference_matrix()
        self.I_r = torch.eye(self.rank)


        for idx in range(self.Block_num):
            Updating_W = Updating_W_Layer(self.rows,self.rank,self.params[f'lambda_tr{idx}'],self.max_iter,self.alpha,self.params[f'K1{idx}'],self.I_r)
            setattr(self, f'Updating_W{idx}', Updating_W)
            Updating_H = Updating_H_Layer(self.cols,self.rank,self.params[f'lambda_pt{idx}'],self.max_iter,self.alpha,self.TTt,self.params[f'K2{idx}'],self.I_r)
            setattr(self, f'Updating_H{idx}', Updating_H)

        self.Updating_A = Updating_A_Layer(self.X, self.epsilon)





    def temporal_difference_matrix(self):
        c = torch.cat([torch.tensor([-1, 2]), torch.zeros(self.t0 - 3), torch.tensor([-1]), torch.zeros(self.cols - self.t0)])
        r = torch.cat([torch.tensor([-1]), torch.zeros(self.cols - self.t0)])
        T = toeplitz(c, r)
        T = torch.tensor(T, dtype=torch.float32)
        TTt = torch.mm(T, T.T)
        TTt = torch.tensor(TTt,dtype=torch.float32)
        return TTt



    def forward(self,Omega,W,H,A,lambda_a,L,i):

        for idx in range(self.Block_num):
            W_pre = W.clone()
            H_pre = H.clone()
            Z = self.X[i] * Omega - A
            W = getattr(self, f'Updating_W{idx}')(Omega, W_pre, H, L, Z)
            H = getattr(self, f'Updating_H{idx}')(Omega, W, H_pre, Z)
            A, lambda_a = self.Updating_A(Omega, W, H, lambda_a,i)


        return W,H,A,lambda_a


class Updating_W_Layer(nn.Module):
    def __init__(self,rows,rank,lambda_tr,max_iter,alpha,K1,I_r):
        super(Updating_W_Layer,self).__init__()
        self.rows = rows
        self.rank = rank
        self.lambda_tr = lambda_tr
        self.max_iter = max_iter
        self.alpha = alpha
        self.K1 = K1
        self.I_r = I_r



    def forward(self,Omega,W_pre,H,L,Z):

        L_W = torch.zeros(self.rows, self.rank, self.rank).to(device)
        V_W = torch.zeros(self.rows, 1, self.rank).to(device)
        lambda_tr = self.lambda_tr.clone()

        for i in range(Omega.shape[0]):
            mask = Omega[i, :].bool()
            H_I = H[:, mask]
            c_I = Z[i, mask].unsqueeze(0)
            V_W[i] = c_I @ H_I.T
            L_W[i] = H_I @ H_I.T + self.alpha * self.I_r

        V_W = V_W.squeeze(1)
        eigvals = torch.linalg.eigvals(L_W)
        lambda_W = eigvals.real.max(dim=1).values.unsqueeze(1)
        tw = 1
        termination_condition_w = 1

        while (termination_condition_w > 1e-8) and (tw < self.max_iter):
            W = W_pre.clone()
            L_WmulW = torch.matmul(W.unsqueeze(1), L_W).squeeze(1)
            Q_WmulW = W * lambda_W
            R = 2 * (L_WmulW - Q_WmulW + lambda_tr * (L @ W_pre) - V_W) / lambda_W
            W_pre = torch.relu((-1 / 2) * R)
            termination_condition_w = (torch.norm(W_pre - W, 2) ** 2) / (self.rank * self.rows)
            tw += 1

        W = W_pre.clone()
        L_WmulW = torch.matmul(W.unsqueeze(1), L_W).squeeze(1)
        Q_WmulW = W * lambda_W
        R = 2 * (L_WmulW - Q_WmulW + self.lambda_tr.clone() * (L @ W_pre) - V_W) / lambda_W
        W_pre = self.K1.clone() * torch.relu((-1 / 2) * R)

        return W_pre

class Updating_H_Layer(nn.Module):
    def __init__(self,cols,rank,lambda_pt,max_iter,alpha,TTt,K2,I_r):
        super(Updating_H_Layer,self).__init__()
        self.cols = cols
        self.rank = rank
        self.lambda_pt = lambda_pt
        self.max_iter = max_iter
        self.alpha = alpha
        self.TTt = TTt
        self.K2 = K2
        self.I_r = I_r


    def forward(self,Omega,W,H_pre,Z):

        L_H = torch.zeros(self.cols, self.rank, self.rank).to(device)
        U_H = torch.zeros(self.cols, self.rank, 1).to(device)
        lambda_pt = self.lambda_pt.clone()

        for j in range(Omega.shape[1]):
            mask = Omega[:, j].bool()
            W_I = W[mask, :]
            b_I = Z[mask, j].unsqueeze(1)
            U_H[j] = W_I.T @ b_I
            L_H[j] = W_I.T @ W_I + self.alpha * self.I_r

        U_H = U_H.squeeze(-1).permute(1, 0)
        eigvals = torch.linalg.eigvals(L_H)
        lambda_H = eigvals.real.max(dim=1).values.unsqueeze(0)
        th = 1
        termination_condition_h = 1

        while (termination_condition_h > 1e-8) and (th < self.max_iter):
            H = H_pre.clone()
            L_HmulH = torch.matmul(L_H, H.T.unsqueeze(-1)).squeeze(-1).T
            Q_HmulH = lambda_H * H
            S = 2 * (L_HmulH - Q_HmulH + lambda_pt * (H_pre @ self.TTt) - U_H) / lambda_H
            H_pre = torch.relu((-1 / 2) * S)
            termination_condition_h = (torch.norm(H_pre - H, 2) ** 2) / (self.rank * self.cols)
            th += 1

        H = H_pre.clone()
        L_HmulH = torch.matmul(L_H, H.T.unsqueeze(-1)).squeeze(-1).T
        Q_HmulH = lambda_H * H
        S = 2 * (L_HmulH - Q_HmulH + self.lambda_pt.clone() * (H_pre @ self.TTt) - U_H) / lambda_H
        H_pre = self.K2.clone() * torch.relu((-1 / 2) * S)

        return H_pre

class Updating_A_Layer(nn.Module):
    def __init__(self,X,epsilon):
        super(Updating_A_Layer,self).__init__()
        self.X = X
        self.epsilon = epsilon

    def forward(self,Omega,W,H,lambda_a,i):

        X_Omega = self.X[i] * Omega
        A = (X_Omega - W @ H) * Omega
        n_1 = A[Omega == 1]
        n_abs = torch.abs(A[Omega == 1])
        n_std = torch.std(n_abs)
        n_sort, _ = torch.sort(n_abs)
        IQR = torch.quantile(n_sort, 0.75) - torch.quantile(n_sort, 0.25)
        deta2 = 1.06 * min(n_std, IQR / 1.34) * len(n_abs) ** -0.2
        w = torch.exp(-(n_abs / deta2))
        anomalies_idx = w < self.epsilon
        anomalies = n_abs[anomalies_idx]
        w_isempty = anomalies.numel() == 0
        if not w_isempty:
            lambda_a = min(torch.min(anomalies ** 2), lambda_a)
        n_1[n_abs < torch.sqrt(lambda_a)] = 0
        A[Omega == 1] = n_1

        return A,lambda_a
