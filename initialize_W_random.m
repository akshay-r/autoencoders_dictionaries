%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function initializes the weights of the autoencoder away from    %%
%% the ground truth A_star. Each row of W is initialized at a random     %%
%% distance away from the corresponding column of A_star                 %%
%% Inputs:                                                               %%
%%        A_star - ground truth dictionary                               %%
%% Outputs:                                                              %%
%%        W, W_T - initialized parameter value, transpose                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, W_T] = initialize_W_random(A_star)

    dW = randn(size(A_star));
    h = size(A_star,2);
    dW = normc(dW)*diag(randi(25, h,1)+1);
    W_T = A_star + dW;
    W = W_T';

end
