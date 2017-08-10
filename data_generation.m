%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function generates a random incoherent dictionary (A_star)        %%
%% and generates signals from random sparse combinations of its columns   %%
%%                                                                        %%
%% Inputs:                                                                %%
%%        n - dimension of signals Y                                      %%
%%        h - sparse signal dimension, dictionary is of size n x h        %%
%%        k - sparsity level                                              %%
%%        num_datapoints - total number of signals to generate, 95-5      %%
%%                         train-test split                               %%
%%        m1 - mean of the generative distribution for the sparse coeffs  %%
%% Outputs:                                                               %%
%%        X,Y - training dataset of sparse coeffs and signals             %%
%%        X_test, Y_test - test dataset                                   %%
%%        A_star - Ground Truth dictionary from which signals were        %%
%%                 generated                                              %%
%%        coherence - coherence of the dictionary A_star                  %%
%%        m2 - second moment of the generative distribution for           %%
%%             sparse coeffs                                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, Y, X_test, Y_test, A_star, coherence, m2] = data_generation(n, h, k, num_datapoints, m1)
    A_star = randn(n,h);
    A_star = normc(A_star);
    coherence_mat = A_star'*A_star;
    coherence = max(max(abs(coherence_mat - eye(h))))/sqrt(n);

    num_test = ceil(0.05*num_datapoints);
    num_train = num_datapoints - num_test;

    X = zeros(h,num_train);
    X_test = zeros(h,num_test);

%     var_x_star = 1/(h*log(n));
    var_x_star = 1/256;
    X_k = normrnd(m1, var_x_star, [k num_datapoints]);

    for i=1:num_train
        supp = sort(randperm(h,k));
        X(supp,i) = X_k(:,i);
    end
    for i=1:num_test
        supp = sort(randperm(h,k));
        X_test(supp,i) = X_k(:,num_train+i);
    end
    Y = A_star*X;
    Y_test = A_star*X_test;

    m2 = var_x_star + m1^2;

end
