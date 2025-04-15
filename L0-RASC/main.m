clear

%% inputdata
data=cell2mat(struct2cell(load('sz_data.mat')));
OD=cell2mat(struct2cell(load('sz_OD.mat')));
X=data./max(data);

data_Size=size(data);

%% parameters initialization
clear options;
options.lambda_tr=0.1;
options.lambda_pt=0.5;
options.lambda_a=1000;
options.alpha=0.1;
options.maxiter=500;
options.epsilon=1e-35;
options.tol=1e-3;
options.r=4;
options.lossprob=0.2;
options.t0=288;

disp(options);

%% random missing data
Omega = ones(data_Size);
Omega(randsample(prod(data_Size), fix(options.lossprob*prod(data_Size)))) = 0;
X_Omega = X.* Omega;


%% rank adaptive
[options.r] = adaptiveR(X_Omega,Omega);

%% model
[W,H,A_Omega, MSE_iters]=l0RASC(X,X_Omega,Omega,OD,options); 
figure
semilogy(MSE_iters,'k--','LineWidth',1.2);
xlabel('Iteration number');
ylabel('MSE');


%% evaluate
[Recovermatrix,NMAE,MSE] = measurement(X,Omega,W,H);
disp(NMAE);
disp(MSE);
disp(options.r);







