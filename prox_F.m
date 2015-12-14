function M = prox_F(W)
    [V D] = eig(W);
    d = diag(D);
    d(d<0)=0;
    M = V*diag(d)*V';
end
