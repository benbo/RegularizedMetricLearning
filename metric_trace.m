function [M,k,loss] = metric_trace(X,Y,Ytil,stepsize,max_it,lm,eps,M)
    [n,m] = size(X);
    loss_last = l_loss(X,Y,Ytil,M,n)+lm*sum(svd(M));
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m);
        Mup = prox_tr(M-stepsize*V,stepsize*lm);
        Mup = (Mup+Mup')/2;
        loss = l_loss(X,Y,Ytil,Mup,n)+lm*sum(svd(Mup));
        if loss_last-loss<eps
            if loss_last>loss
                M=Mup;
            else
                %minimum overshot, set variables to previous value
                loss = loss_last;
                k=k-1;
            end
            break
        end
        M=Mup;
        loss_last = loss
    end 
end

