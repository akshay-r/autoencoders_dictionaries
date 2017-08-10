%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function initializes the weights of the autoencoder away from    %%
%% the ground truth A_star. Each row of W is initialized at a fixed      %%
%% distance (delta) away from the corresponding column of A_star         %%
%% Inputs:                                                               %%
%%        A_star - ground truth dictionary                               %%
%%        delta - distance at which to initialize dictionary             %%
%% Outputs:                                                              %%
%%        W, W_T - initialized parameter value, transpose                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, W_T] = initialize_W(A_star,delta)
    dW = randn(size(A_star));
    dW = delta*normc(dW);

    W_T = A_star + dW;
    W = W_T';
end
