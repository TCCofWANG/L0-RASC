function [Recovermatrix,NMAE,MSE] = measurement(TM,Omega,W,H)

    Recovermatrix=W*H;
     NMAE=sumabs((Recovermatrix-TM).*(1-Omega))/sumabs(TM.*(1-Omega));%比较缺失处的数据
    MSE=(norm(Recovermatrix-TM,2))/numel(TM);

end

