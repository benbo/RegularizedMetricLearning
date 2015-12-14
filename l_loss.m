function v = l_loss(X,Y,Ytil,M,n)
    v = 0;
    for row = 1:n
        v = v + log_one_x_exp(Ytil(row)-Y(row)*X(row,:)*M*X(row,:)');
    end
end
