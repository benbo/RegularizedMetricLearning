function [M,k,loss] = metric_trace_transfer(X,Y,Ytil,stepsize,max_it,lm,eps,alpha,Mt,M)
    [n,m] = size(X);
    loss_last = l_loss(X,Y,Ytil,M,n)+lm*sum(svd(M)) + alpha *norm(M-Mt,'fro')^2;
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m) + 2*alpha*(M-Mt);
        Mup = prox_tr(M-stepsize*V,stepsize*lm);
        Mup = (Mup+Mup')/2;
        loss = l_loss(X,Y,Ytil,Mup,n)+lm*sum(svd(Mup)) + alpha *norm(Mup-Mt,'fro')^2;
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

