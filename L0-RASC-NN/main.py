import os
import torch
import time
from utils.utils import Loss
from utils.utils import init_Dataset
from network.L0_RASC_NN import L0RANMFNN_Layer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchsummary import summary

if __name__ == '__main__':

    para,X,train_data,test_data = init_Dataset()
    model = L0RANMFNN_Layer(X,para).to(device)
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = Loss().to(device)
    print("start training...")

    for epoch in range(1):
        for i in range (para['training set size']):
            W_pre = train_data['W'][i]
            H_pre = train_data['H'][i]
            A_pre = train_data['A'][i]
            lambda_a_pre = train_data['lambda_a'][i]
            L = train_data['L'][i]
            Omega = train_data['Omega'][i]
            optimizer.zero_grad()
            iter_time_start = time.time()

            W,H,A,lambda_a = model(Omega, W_pre, H_pre, A_pre, lambda_a_pre, L,i)
            iter_time_end = time.time()
            loss_normal = criterion(W @ H * (1 - Omega), X[i] * (1 - Omega))
            loss_normal.backward(retain_graph=True)
            time1 = time.time()
            optimizer.step()
            time2 = time.time()
            nmae = torch.sum(torch.abs((X[i] - W @ H) * (1 - Omega))) / torch.sum(torch.abs(X[i] * (1 - Omega)))
            residual_err = torch.norm((W @ H + A - W @ H_pre - A_pre) * Omega, 'fro') / torch.norm((W @ H_pre + A_pre) * Omega, 'fro')
            iter_time = time2 - iter_time_start
            print(f'\n [epoch:{epoch + 1}][{i + 1}/{para["training set size"]}] residual error:{residual_err.item():.4f} NMAE:{nmae.item():.4f} Time：{iter_time:.4f}s')
            train_data['W'][i] = W
            train_data['H'][i] = H
            train_data['A'][i] = A
            train_data['lambda_a'][i] = lambda_a
    model.eval()
    print("Training complete")
    print("start testing...")
    with torch.no_grad():
        for epoch in range(1):
            for i in range(para['testing set size']):
                W_pre = test_data['W'][i]
                H_pre = test_data['H'][i]
                A_pre = test_data['A'][i]
                lambda_a_pre = test_data['lambda_a'][i]
                L = test_data['L'][i]
                Omega = test_data['Omega'][i]
                iter_time_start = time.time()
                W, H, A, lambda_a = model(Omega, W_pre, H_pre, A_pre, lambda_a_pre, L,i+para['training set size'])
                iter_time_end = time.time()
                nmae = torch.sum(torch.abs((X[i] - W @ H) * (1 - Omega))) / torch.sum(torch.abs(X[i] * (1 - Omega)))
                residual_err = torch.norm((W @ H + A - W @ H_pre - A_pre) * Omega, 'fro') / torch.norm((W @ H_pre + A_pre) * Omega, 'fro')
                iter_time = iter_time_end - iter_time_start
                print(f'\n [epoch:{epoch + 1}][{i + 1}/{para["training set size"]}] residual error:{residual_err.item():.4f} NMAE:{nmae.item():.4f} Time：{iter_time:.4f}s')
                test_data['W'][i] = W
                test_data['H'][i] = H
                test_data['A'][i] = A
                test_data['lambda_a'][i] = lambda_a
    print("Testing complete")