function M = prox_tr(W,s)
    [V D] = eig(W);
    d = diag(D)-s;
    d(d<0)=0;
    M = V*diag(d)*V';
end
