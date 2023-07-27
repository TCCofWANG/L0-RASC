function R = adaptiveR(X,Omega)

[N,M]=size(X);

S=zeros(N,N);
for i=1:N
       
        compared_posi=repmat(Omega(i,:),[N,1]).*Omega;
        X_compared=X.*compared_posi;
        X_i=repmat(X(i,:),[N,1]).*compared_posi;

        X_comp_mean=sum(X_compared,2)./sum(compared_posi,2);
        X_comp_std=sqrt(sum((X_compared-repmat(X_comp_mean,[1,M])).^2,2)./sum(compared_posi,2));
        X_compared_nor=((X_compared-repmat(X_comp_mean,[1,M]))./repmat(X_comp_std,[1,M])).*compared_posi;

        X_i_mean=sum(X_i,2)./sum(compared_posi,2);
        X_i_std=sqrt(sum((X_i-repmat(X_i_mean,[1,M])).^2,2)./sum(compared_posi,2));
        X_i_nor=((X_i-repmat(X_i_mean,[1,M]))./repmat(X_i_std,[1,M])).*compared_posi;


       S(:,i)=sqrt(sum((X_i_nor-X_compared_nor).^2,2));
end
s_std=std(S,0,"all");
deta=1.06*s_std*(N^(-0.2));
S=(1-eye(N)).*exp(-((S.^2)/(2*(deta^2))));

DS_lr=diag((sqrt(sum(S,2))).^(-1));
L=eye(N)-DS_lr*S*DS_lr;

eigofL=eig(L);
eigofL=real(eigofL);
descendeig=sort(eigofL,'descend');
eigdiff=-diff(descendeig);
[~, locs]=findpeaks(eigdiff);
if isempty(locs)==1
   locs=5;
end
R=locs(1);
end

