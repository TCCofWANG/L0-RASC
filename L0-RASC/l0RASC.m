function [W,H,A_Omega, MSE] = l0RASC(X,X_Omega,Omega,OD,options)

[rows,cols]=size(X);
rank=options.r;
lambda_tr=options.lambda_tr;
lambda_pt=options.lambda_pt;
lambda_a=options.lambda_a;
alpha=options.alpha;
maxiter=options.maxiter;
epsilon=options.epsilon;
tol=options.tol;
t0=options.t0;

MSE = zeros(1, maxiter);

%% Laplacian matrix
Similarity=zeros(rows,rows);
for i=1:rows
    for j=i+1:rows
       Pm=(Omega(i,:)&Omega(j,:));
       Xi_Pm=X_Omega(i,:).*Pm;
       Xi_Pm(Pm==0)=[];
       Xj_Pm=X_Omega(j,:).*Pm;
       Xj_Pm(Pm==0)=[];

       if isempty(Xj_Pm)
           Similarity(i,j)=0;
       else
           Similarity(i,j)=dot(Xi_Pm,Xj_Pm)/(norm(Xi_Pm,2)*norm(Xj_Pm,2));
       end
    end
    Similarity(i,i)=0;
end

S=(Similarity+Similarity').*OD;
L=diag(sum(S,2))-S;

%% temporal difference matrix
c=[-1,2,zeros(1,t0-1),-1,zeros(1,cols-2-t0)];
r=[-1,zeros(1,cols-2-t0)];
T=toeplitz(c,r);
TTt=T*T';

%% initialize matrices
% factor matrices
rng(9);
W = randn(rows,rank);
H = rand(rank,cols);

% anomaly matrix
x = X_Omega(Omega==1);
x_sort = sort(x);
IQR = (prctile(x_sort-mean(x),75) - prctile(x_sort-mean(x),25));
deta2 = 1.06*min(std(x),IQR/1.34)*length(x)^-0.2;
w = exp(-(abs(x-mean(x))./(deta2)));
anomalies = X_Omega(Omega(w<epsilon)==1);
A_Omega = zeros(rows,cols);
A_Omega(Omega(w<epsilon)==1) = anomalies;

I_r=eye(rank);%identity matrix

%% Update process
residual_err2=1;k=0;

while (residual_err2>tol) && (k<=maxiter)
    
    k=k+1;
    W_pre=W; H_pre=H; A_pre=A_Omega;


    Z=X_Omega-A_Omega;    
    
    % update W
    for i=1:rows
        w=W_pre(i,:);

        col = (Omega(i,:) == 1);
        H_I = H(:,col);
        c_I = Z(i,col);

        L_H=H_I*H_I'+alpha*I_r;
        L_H_eig_max=max(eig(L_H));
        Q_H=L_H_eig_max*I_r;

        % update w_i^t
        tw=1; termination_condition_w=1;

        while (termination_condition_w>1e-8)&&(tw<50)
            wt=w;
            q_w=(2/L_H_eig_max)*((wt*(L_H-Q_H))-c_I*H_I'+lambda_tr*L(i,:)*W_pre);
            w(q_w<0)=(-1/2)*q_w(q_w<0);
            w(q_w>=0)=0;
            termination_condition_w=(norm(w-wt,2)^2)/(rank);
            tw=tw+1;
            W(i,:)=w;
        end
        
    end

    % update H
    for j=1:cols
        h=H_pre(:,j);
        row = (Omega(:,j) == 1);
        W_I =  W(row,:);
        b_I=Z(row,j);

        L_W=W_I'*W_I+alpha*I_r;
        L_W_eig_max=max(eig(L_W));
        Q_W=L_W_eig_max*I_r;

        % update h_j^t
        th=1;termination_condition_h=1;
        while (termination_condition_h>1e-8)&&(th<50)
            ht=h;
            p_h=(2/L_W_eig_max)*(((L_W-Q_W)*ht)-W_I'*b_I+lambda_pt*H_pre*TTt(:,j));
            h(p_h<0)=(-1/2)*p_h(p_h<0);
            h(p_h>=0)=0;
            termination_condition_h=(norm(h-ht,2)^2)/(rank);
            th=th+1;
            H(:,j)=h;
        end
        
    end

    % update A
    A_Omega=(X_Omega-W*H).*Omega;
    n_1=A_Omega(Omega==1);
    n_abs=abs(A_Omega(Omega==1));

    % update lambda_a
    n_std=std(n_abs);
    n_sort=sort(n_abs);
    IQR = (prctile(n_sort,75) - prctile(n_sort,25));
    deta2 = 1.06*min(n_std,IQR/1.34)*length(n_abs)^-0.2;
    w = exp(-(n_abs./(deta2)));
    anomalies_idx = (w<epsilon); 
    anomalies=n_abs(anomalies_idx);
    w_isempty=isempty(anomalies);
    if ~w_isempty
        lambda_a=min(min(anomalies.^2),lambda_a);
    end
    n_1(n_abs<sqrt(lambda_a))=0;  
    A_Omega(Omega==1)=n_1; 

    % termination condiction
    mse=norm(X-W*H,'fro').^2/(rows*cols);
    MSE(1,k)=mse;

    residual_err=norm((W_pre*H_pre+A_pre-W*H-A_Omega).*Omega,'fro')/norm((W_pre*H_pre+A_pre).*Omega,'fro');

    NMAE1=sumabs((W*H-X).*(1-Omega))/sumabs(X.*(1-Omega));

    fprintf('\n iter:%d, residual error:%f, MSE:%f',k,residual_err,mse);
    if NMAE1>10
        break
    end
         
end

end
