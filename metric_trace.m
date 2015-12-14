function [M,k] = metric_trace(X,Y,Ytil,stepsize,max_it,lm,eps)
    [n,m] = size(X);
    %initialize metric matrix M
    M=ones(m);
    loss_last = Inf;
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m);
        M = prox_tr(M-stepsize*V,stepsize*lm);
        %symmtrize since roundoff errors can lead to result not being symmetric
        M = (M+M')/2;
        loss = l_loss(X,Y,Ytil,M,n)+lm*sum(svd(M));
        if loss_last-loss<eps
            break
        end
        loss_last = loss;
    end 
end

