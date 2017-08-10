n = 100; num_datapoints = 5000;
% H = [256, 512, 1024];
% H = [1024];
H = [2048, 4096];
P = [0.2, 0.3, 0.4, 0.5];
Y_diff_init_norm = zeros(length(H), length(P));
Y_diff_final_norm = zeros(length(H), length(P));

W_reps = 3;
A_reps = 3;

for i=1:length(H)
    for j=1:length(P)
        h = H(i); p = P(j); k = ceil(h^p); m1 = -1*h^(-3/2);
        
        sprintf('Hidden dimension: %d',h)
        
        for u=1:A_reps
            [X, Y, X_test, Y_test, A_star, coherence, m2] = data_generation(n, h, k, num_datapoints, m1);
            delta = 0.80;
            eta = 0.9;
            epsilon_i = 1/2*abs(m1)*k*(delta+coherence); % epsilon in theorem 3.1
            threshold = 1e-8;
            max_iter = 100;

            for v=1:W_reps
%                 init_delta = 2.0;
%                 [W, W_T] = initialize_W(A_star, init_delta);
                [W, W_T] = initialize_W_random(A_star);
                W0 = W;

                Y_diff_init = W_T*X_test - Y_test;
                Y_diff_init_norm(i,j) = Y_diff_init_norm(i,j) + sum(sqrt(sum(Y_diff_init.^2,1)))/size(Y_test,2);

                [W_final, final_norm] = grad_descent(W, X, Y, k, eta, delta, epsilon_i, threshold, max_iter);
                sprintf('Final Gradient Norm: %.8f',final_norm)

                W_final_norm_T = normc(W_final');
                Y_diff_final = W_final_norm_T*X_test - Y_test;
                Y_diff_final_norm(i,j) = Y_diff_final_norm(i,j) + sum(sqrt(sum(Y_diff_final.^2,1)))/size(Y_test,2);

                diff = W_final' - A_star;
                diff_norm = zeros(size(A_star,2),1);
                for t=1:size(diff,2)
                    diff_norm(t) = norm(diff(:,t),2);
                end

                init_diff = W0' - A_star;
                init_diff_norm = zeros(size(A_star,2),1);
                for t=1:size(init_diff,2)
                    init_diff_norm(t) = norm(init_diff(:,t),2);
                end
                
                save(char('result_'+string(h)+'_'+string(p)+'_'+'u'+string(u)+'_'+'v'+string(v)+'.mat'));
            end
        end
        Y_diff_init_norm(i,j) = Y_diff_init_norm(i,j)/(W_reps*A_reps);
        Y_diff_final_norm(i,j) = Y_diff_final_norm(i,j)/(W_reps*A_reps);
    end
end


save(char('result_final.mat'));
