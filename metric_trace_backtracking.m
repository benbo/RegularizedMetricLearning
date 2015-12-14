function [M,k] = metric_trace(X,Y,Ytil,stepsize,max_it,lm,eps,M)
    [n,m] = size(X);
    %initialize metric matrix M
    beta=0.1;
    loss_last = Inf;
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m);
        M = prox_tr(M-stepsize*V,stepsize*lm);
        %t=1;
        %Gt = (M-prox_tr(M-t*V,t*lm));
        %g=l_loss(X,Y,Ytil,M,n);
        %while l_loss(X,Y,Ytil,M-Gt,n) > g-sum(sum(V.*Gt))+t/2*norm(Gt/t,'fro')
        %    t=t*beta;
        %    Gt = (M-prox_tr(M-t*V,t*lm));
        %end
        %symmtrize since roundoff errors can lead to result not being symmetric
        %M=-Gt+M;
        M = (M+M')/2;
        loss = l_loss(X,Y,Ytil,M,n)+lm*sum(svd(M));
        if loss_last-loss<eps
            break
        end
        %t
        loss_last = loss
    end 
end

