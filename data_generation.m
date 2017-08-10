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
