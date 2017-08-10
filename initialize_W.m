function [W, W_T] = initialize_W(A_star,delta)
    dW = randn(size(A_star));
    dW = delta*normc(dW);

    W_T = A_star + dW;
    W = W_T';
end
