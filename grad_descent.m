function [W_final, final_norm] = grad_descent(W, X, Y, k, eta, delta, epsilon_i, threshold, max_iter)
    grad_norm = zeros(max_iter,1);
    W_init = W;
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