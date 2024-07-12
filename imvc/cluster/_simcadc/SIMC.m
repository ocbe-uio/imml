function [UU,V,A,W,Z_final,iter,obj] = SIMC(Y,num_sample,lambda,d,num_anchor,N,Ne,E,beta,gamma,eps)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di
% N      : ni*n

%% initialize
maxIter = 50 ; % the number of iterations

num_view = length(Y);

W = cell(num_view,1);            % di * d
A = zeros(d,num_anchor);         % d  * m
Z_final = zeros(num_anchor,num_sample); % m  * n
% Z_final_copy = Z_final;

for i = 1:num_view
   Y{i} = zscore(Y{i},0,1)'; % turn into d*n
   di = size(Y{i},1); 
   W{i} = zeros(di,d);
   NNT{i} = N{i}*N{i}';
   B{i} = zeros(di,num_sample);
   Z{i} = zeros(num_anchor,num_sample);
   R{i} = eye(num_anchor);
end
Z_final(:,1:num_anchor) = eye(num_anchor);


alpha = ones(1,num_view)/num_view;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% update X
    for iv = 1:num_view
        X{iv} = Y{iv} + E{iv} * N{iv};
    end
    
    %% update W_i
    for iv = 1:num_view
        AZ = A*Z{iv};
        C = X{iv}*AZ';      
        [U,~,V] = svd(C,'econ');
        W{iv} = U*V';
    end

    %% update A
    part1 = 0;
    for ia = 1:num_view
        al2 = alpha(ia)^2;
        part1 = part1 + al2 * W{ia}' * X{ia} * Z{ia}';
    end
    [Unew,~,Vnew] = svd(part1,'econ');
    A = Unew*Vnew';

    %% update Z_i
    for iv = 1:num_view
        C1 = alpha(iv)^2 * X{iv}'*W{iv}*A + gamma * Z_final'* R{iv};
        C2 = alpha(iv)^2 + gamma;
        C1 = C1';
        for ii = 1:num_sample
            ut = C1(:,ii)/C2;
            Z{iv}(:,ii) = EProjSimplex_new(ut);
        end
    end

    %% update Z_final
    C3 = 0;
    for iv = 1:num_view
        C3 = C3 + gamma * Z{iv}'*R{iv}';
    end
    C3 = C3';
    C4 = num_view * gamma + 1;
    for ii = 1:num_sample
        ut = C3(:,ii)/C4;
        Z_final(:,ii) = EProjSimplex_new(ut);
    end

    %% update E
    for iv = i:num_view
        B{iv} = Y{iv} - W{iv} * A * Z_final;
        E{iv} = -B{iv}*N{iv}'/(NNT{iv}+beta * ones(Ne(iv)));
    end

    %% update R
    for iv = 1:num_view
        [U_R, ~, V_R] = svd(Z{iv} * Z_final','econ');
        R{iv} = U_R * V_R';
    end

    %% update alpha
    for iv = 1:num_view
        avoiding_zero = norm(B{iv}+E{iv}*N{iv},'fro') + eps;
        alpha(iv)=sqrt(1/avoiding_zero);
    end

    %%
    term1 = 0;
    for iv = 1:num_view
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A * Z_final,'fro')^2;
    end
    term2 = lambda * norm(Z_final,'fro')^2;
    obj(iter) = term1+ term2;


    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(Z_final','econ');
        flag = 0;
    end
end
UU = UU ./ repmat(sqrt(sum(UU.^2, 2)), 1,size(UU,2));
end

    
