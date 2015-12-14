function [M,k] = metric_frob(X,Y,Ytil,stepsize,max_it,lm,eps)
    [n,m] = size(X);
    %initialize metric matrix M
    M=zeros(m);
    loss_last = Inf;
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m)+2*lm*M;
        M = prox_F(M-stepsize*V);%projection onto psd cone
        M = (M+M')/2;%symmtrize since roundoff errors can lead to result not being symmetric
        loss = l_loss(X,Y,Ytil,M,n)+lm*norm(M,'fro')^2;
        if loss_last-loss<eps
            break
        end
        loss_last = loss
    end 
end

