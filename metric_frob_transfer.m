function [M,k] = metric_frob_transfer(X,Y,Ytil,stepsize,max_it,lm,eps,alpha,Mt,M)
    [n,m] = size(X);
    loss_last = Inf;
    for k = 1:max_it
        V = l_grad(X,Y,Ytil,M,n,m)+2*lm*M+ 2*alpha*(M-Mt);
        Mup = prox_F(M-stepsize*V);%projection onto psd cone
        Mup = (M+M')/2;%symmtrize since roundoff errors can lead to result not being symmetric
        loss = l_loss(X,Y,Ytil,Mup,n)+lm*norm(Mup,'fro')^2 + alpha *norm(Mup-Mt,'fro')^2;
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

