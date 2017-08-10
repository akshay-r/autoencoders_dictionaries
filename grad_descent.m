%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This function performs full batch gradient descent on our proposed    %%
%% autoencoder loss function. GD is done with a constant step size       %%
%% unless the gradient norm increases, in which case it is reduced and   %%
%% the iterations continue. GD terminates when gradient norm falls below %%
%% a certain factor of the initial gradient norm.                        %%
%% Inputs:                                                               %%
%%        W_init - initial parameter value                               %%
%%        X,Y - training data, sparse coeffs and signals                 %%
%%        k - number of non-zero coeffs                                  %%
%%        eta - learning rate                                            %%
%%        delta - parameter as defined in the paper                      %%
%%        epsilon_i - bias for ReLU hidden layer                         %%
%%        threshold - for terminating gradient descent                   %%
%%        max_iter - maximum number of iterations to run GD for          %%
%% Outputs:                                                              %%
%%        W_final - final parameter value                                %%
%%        final_norm - gradient norm at W_final                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W_final, final_norm] = grad_descent(W_init, X, Y, k, eta, delta, epsilon_i, threshold, max_iter)
    grad_norm = zeros(max_iter,1);
    W = W_init;
    iter = 1;

    while iter <= max_iter
        grad_mat = grad(W,X,Y,k,delta,epsilon_i);
        % norm of gradient of each row, at every iteration
        grad_norm(iter) = norm(grad_mat, 'fro');
%         display([iter grad_norm(iter)]);
        W = W-eta*grad_mat; % updating W matrix

        if iter==2
            if grad_norm(iter) > grad_norm(1)
                disp('Reducing learning rate and restarting')
                eta = eta/3;
                W = W_init;
                iter = 0;
                grad_norm = zeros(max_iter,1);
            end
        end

        if (iter>2)
            if grad_norm(iter) > grad_norm(iter-1)
                disp('changing learning rate')
                eta = eta/3;
            end
        end

        if (iter>1)
            if (grad_norm(iter) <= grad_norm(1)*threshold)
                break;
            end
        end
        iter = iter + 1;
    end
    final_norm = grad_norm(find(grad_norm >0));
    final_norm = final_norm(end);
    W_final = W;
end
