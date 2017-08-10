function grad_mat = grad(W, X, Y, k, delta, epsilon_i)
    h = size(X,1);
    n = size(Y,1);
    N = size(Y,2);
    
    C = (1 - delta)^2;  % remark, after theorm 3.2
    q_i = k/h; % sparsity probability  
    lambda_1 = C*h*k + h*q_i*(1 - delta)^2;% 3.2 proposed gradient, lambda 1
    lambda_2 = -1; % 3.2 proposed gradient, lambda 1
    
    grad_mat = zeros(h,n);
    W_T = W';

    for j= 1:N
        supp = sort(find(X(:,j)~=0));
        W_tilde = W(supp,:);
        for i=1:k
            matrix_factor = ((W_T(:,supp(i))'*Y(:,j) - epsilon_i).*eye(n) + (W_T(:,supp(i))*Y(:,j)'));
            vector_factor = W_T*max(W*Y(:,j) - epsilon_i,0) - C*h*Y(:,j);
            squared_loss_grad = matrix_factor*vector_factor;

            weight_regularization_grad  = lambda_1*W_T(:,supp(i));% first regularization term

            activity_regularization_grad = lambda_2*( (norm(W_tilde*Y(:,j),2)^2) * W_T(:,supp(i)) + (norm(W_tilde, 'fro')^2) * (W_T(:,supp(i))'*Y(:,j)) *Y(:,j));

            grad_mat(supp(i),:) = grad_mat(supp(i),:) + (squared_loss_grad + weight_regularization_grad + activity_regularization_grad)'; 
        end
    end
    grad_mat = (1/N)*grad_mat;
end