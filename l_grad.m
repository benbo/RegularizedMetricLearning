function grad = l_grad(X,Y,Ytil,M,n,m)
    grad = zeros(m);
    for row = 1:n
        grad = grad - Y(row)*X(row,:)'*X(row,:)/(1+exp(-Ytil(row)+Y(row)*X(row,:)*M*X(row,:)'));
    end
end
