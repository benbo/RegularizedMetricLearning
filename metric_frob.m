function [M,k,loss] = metric_frob(X,Y,Ytil,stepsize,max_it,lm,eps,M)
    [n,m] = size(X);
    loss_last = l_loss(X,Y,Ytil,M,n)++lm*norm(M,'fro')^2;
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m)+2*lm*M;
        Mup = prox_F(M-stepsize*V);%projection onto psd cone
        Mup = (Mup+Mup')/2;%symmtrize since roundoff errors can lead to result not being symmetric
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

