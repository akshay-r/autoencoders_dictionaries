function [W, W_T] = initialize_W_random(A_star)

    dW = randn(size(A_star));
    h = size(A_star,2);
    dW = normc(dW)*diag(randi(25, h,1)+1);
    W_T = A_star + dW;
    W = W_T';

end
