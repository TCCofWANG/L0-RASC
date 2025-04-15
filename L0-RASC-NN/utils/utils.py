import torch
from scipy.signal import find_peaks
import scipy.io
import csv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Getdata():
    X_data = scipy.io.loadmat('data/Portal/Portal_data.mat')
    variable_names = [key for key in X_data.keys() if not key.startswith('__')]
    X = torch.tensor(X_data[variable_names[0]], dtype=torch.float32)
    X = X[:, 1:]
    OD_data = scipy.io.loadmat('data/Portal/Portal_OD.mat')
    variable_names = [key for key in OD_data.keys() if not key.startswith('__')]
    OD = torch.tensor(OD_data[variable_names[0]], dtype=torch.float32)
    return X,OD

def loaddata(failname):
    X = []
    with open(failname, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 0:
                try:
                    X.append([float(val) for val in row])
                except ValueError:
                    print(f"Ignoring row with non-numeric data: {row}")

    tensor_data = torch.tensor(X, dtype=torch.float32)
    return tensor_data

def para_set(X):
    para = {}
    para['rows'], para['cols'] = X.shape
    para['t0'] = 288
    para['cols_'] = para['t0']
    para['lambda_tr'] = 0.1
    para['lambda_pt'] = 0.5
    para['lambda_a'] = 1000
    para['alpha'] = 0.1
    para['max_iter'] = 20
    para['epsilon'] = 1e-35
    para['tol'] = 1e-3
    para['r'] = 4
    para['lossprob'] = 0.3

    para['Block_num'] = 20

    para['training set size'] = int(para['cols'] / para['t0'] - 2)
    para['testing set size'] = 1
    return para

def init_Dataset():
    X,OD = Getdata()
    para = para_set(X)
    chunks = []
    for i in range(int(para['cols'] / para['t0'])):
        start_col = i * para['t0']
        end_col = start_col + para['t0']
        chunk = X[:, start_col:end_col]
        chunks.append(chunk)
    X = chunks
    Data_Size = chunks[0].size()
    initial_seed = 2
    torch.manual_seed(initial_seed)
    train_data = {'Omega': [], 'W':[], 'H': [], 'A': [], 'lambda_a': [], 'L': []}
    test_data = {'Omega': [], 'W':[], 'H': [], 'A': [], 'lambda_a': [], 'L': []}
    print('Load Data...')
    for i in range(para['training set size']):
        Omega = torch.ones(Data_Size)
        num_elements = torch.prod(torch.tensor(Data_Size)).item()
        num_zero_elements = int(para['lossprob'] * num_elements)
        random_indices = torch.randperm(num_elements)[:num_zero_elements]
        Omega.view(-1)[random_indices] = 0
        train_data['Omega'].append(Omega)
        torch.manual_seed(initial_seed + 1)
        if i == 0:
            para['R'] = AdaptiveR(X[i], Omega)

        W = torch.rand(para['rows'], para['R'])
        train_data['W'].append(W)
        H = torch.rand(para['R'], para['cols_'])
        train_data['H'].append(H)
        A = initialize_matrices(X[i],Omega,para)
        train_data['A'].append(A)
        lambda_a = torch.tensor(para['lambda_a'], dtype=torch.float32)
        train_data['lambda_a'].append(lambda_a)
        L = laplacian_matrix(X[i],Omega,para,OD)
        train_data['L'].append(L)

    for i in range(para['testing set size']):
        Omega = torch.ones(Data_Size)
        num_elements = torch.prod(torch.tensor(Data_Size)).item()
        num_zero_elements = int(para['lossprob'] * num_elements)
        random_indices = torch.randperm(num_elements)[:num_zero_elements]
        Omega.view(-1)[random_indices] = 0
        test_data['Omega'].append(Omega)
        torch.manual_seed(initial_seed + 1)

        W = torch.rand(para['rows'], para['R'])
        test_data['W'].append(W)
        H = torch.rand(para['R'], para['cols_'])
        test_data['H'].append(H)
        A = initialize_matrices(X[i],Omega,para)
        test_data['A'].append(A)
        lambda_a = torch.tensor(para['lambda_a'], dtype=torch.float32)
        test_data['lambda_a'].append(lambda_a)
        L = laplacian_matrix(X[i],Omega,para,OD)
        test_data['L'].append(L)

        para = move_to_device(para, device)
        #X = chunks
        X = move_to_device(X, device)
        train_data = move_to_device(train_data, device)
        test_data = move_to_device(test_data, device)
        print('Load Complete')

    return para,X,train_data,test_data

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data

def AdaptiveR(X, Omega):
    N, M = X.shape
    S = torch.zeros((N, N), dtype=torch.float32)

    for i in range(N):
        compared_posi = Omega[i, :].unsqueeze(0).repeat(N, 1) * Omega
        X_compared = X * compared_posi
        X_i = X[i, :].unsqueeze(0).repeat(N, 1) * compared_posi
        X_comp_mean = torch.sum(X_compared, dim=1) / torch.sum(compared_posi, dim=1)
        X_comp_std = torch.sqrt(torch.sum((X_compared - X_comp_mean.unsqueeze(1)) ** 2, dim=1) / torch.sum(compared_posi, dim=1))
        X_compared_nor = ((X_compared - X_comp_mean.unsqueeze(1)) / X_comp_std.unsqueeze(1)) * compared_posi
        X_i_mean = torch.sum(X_i, dim=1) / torch.sum(compared_posi, dim=1)
        X_i_std = torch.sqrt(torch.sum((X_i - X_i_mean.unsqueeze(1)) ** 2, dim=1) / torch.sum(compared_posi, dim=1))
        X_i_nor = ((X_i - X_i_mean.unsqueeze(1)) / X_i_std.unsqueeze(1)) * compared_posi
        S[:, i] = torch.sqrt(torch.sum((X_i_nor - X_compared_nor) ** 2, dim=1))

    s_std = torch.std(S)
    deta = 1.06 * s_std * (N ** (-0.2))
    S = (1 - torch.eye(N)) * torch.exp(-((S ** 2) / (2 * (deta ** 2))))
    DS_lr = torch.diag(torch.sqrt(torch.sum(S, dim=1)) ** (-1))
    L = torch.eye(N) - DS_lr @ S @ DS_lr
    L[torch.isnan(L)] = 0
    eigofL = torch.real(torch.linalg.eigvals(L))
    descendeig, _ = torch.sort(eigofL, descending=True)
    eigdiff = -torch.diff(descendeig)
    eigdiff_np = eigdiff.cpu().numpy()
    peaks, _ = find_peaks(eigdiff_np)
    if len(peaks) == 0:
        R = 5
    else:
        R = peaks[0] + 1


    return R

def initialize_matrices(X,Omega,para):
    x = X[Omega == 1]
    mask_Omega = (Omega == 1)
    x_sort = torch.sort(x).values
    mean_x = torch.mean(x)
    x_centered = x_sort - mean_x
    IQR = torch.quantile(x_centered, 0.75) - torch.quantile(x_centered, 0.25)
    deta2 = 1.06 * min(torch.std(x), IQR / 1.34) * len(x) ** -0.2
    w = torch.ones(para['rows'], para['cols_'])
    w[mask_Omega] = torch.exp(-torch.abs(X[mask_Omega] - mean_x) / deta2)

    mask = (w < para['epsilon'])
    A_Omega = torch.zeros(para['rows'], para['cols_'])
    A_Omega[mask] = X[mask]
    return A_Omega


def laplacian_matrix(X,Omega,para,OD):
    X_Omega = X * Omega
    Similarity = torch.zeros((para['rows'], para['rows']))
    for i in range(para['rows']):
        for j in range(i + 1, para['rows']):
            Pm = (Omega[i, :] == 1) & (Omega[j, :] == 1)
            Xi_Pm = X_Omega[i, :] * Pm
            Xi_Pm = Xi_Pm[Pm != 0]
            Xj_Pm = X_Omega[j, :] * Pm
            Xj_Pm = Xj_Pm[Pm != 0]
            if Xi_Pm.size == 0 or Xj_Pm.size == 0:
                Similarity[i, j] = 0
            else:
                Similarity[i, j] = torch.dot(Xi_Pm, Xj_Pm) / (torch.norm(Xi_Pm, 2) * torch.norm(Xj_Pm, 2))
    Similarity.fill_diagonal_(0)
    S = (Similarity + Similarity.T) * OD
    L = torch.diag(torch.sum(S, dim=1)) - S
    return L

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, target):

        return torch.mean(torch.abs(target - output))