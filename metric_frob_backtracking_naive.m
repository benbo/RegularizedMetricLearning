function [M,k loss] = metric_frob_backtracking_naive(X,Y,Ytil,tstart,max_it,lm,eps,M)
    %naive backtracking, just minimize smooth part while it dominates
    [n,m] = size(X);
    %initialize metric matrix M
    beta=0.1;
    loss_last = l_loss(X,Y,Ytil,M,n)+lm*norm(M,'fro')^2;
    for k = 1:max_it
        t=tstart;
        V = l_grad(X,Y,Ytil,M,n,m) +2*lm*M;
        Z = prox_F(M-t*V,t*lm);
        g=l_loss(X,Y,Ytil,M,n);
        %naive, just minimize smooth part
        while l_loss(X,Y,Ytil,Z,n) > g%+sum(sum(V.*(Z-M)))+t/2*norm(Z-M,'fro')^2
            t=t*beta;
            Z = prox_F(M-t*V,t*lm);
        end
        %symmtrize since roundoff errors can lead to result not being exactly symmetric
        Mup = (Z+Z')/2;
        loss = l_loss(X,Y,Ytil,Mup,n)+lm*norm(Mup,'fro')^2;
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

