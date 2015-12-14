%calculate log(1+exp(x)) and try to avoid overflow errors
function s = log_one_x_exp(x)
    c = x/2; 
    s = c + log(exp(-c) + exp(x-c));
end
