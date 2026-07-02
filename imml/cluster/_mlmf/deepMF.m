function [Hc, H, derror] = deepMF(XX, layers, lamda1, lamda2, z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, bUpdateZ)
% Multi-layer matrix factorization, adapted from the original MLMF MATLAB
% implementation for Octave execution through oct2py.

for iv = 1:length(XX)
    X1 = XX{iv};
    observed = any(X1, 1);
    ind_0 = find(observed == 0);

    W1 = eye(size(X1, 2));
    W1(:, ind_0) = [];
    XX{iv}(:, ind_0) = [];

    G{iv} = W1;
    Ind_ms{iv} = ind_0;
end

numOfView = numel(XX);
num_of_layers = numel(layers);

Z = cell(numOfView, num_of_layers);
H = cell(numOfView, num_of_layers);

for v_ind = 1:numOfView
    X = XX{v_ind};
    X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 1)));

    if ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                V = X;
            else
                V = H{v_ind, i_layer - 1};
            end

            if verbose
                display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', ...
                    i_layer, layers(i_layer), mat2str(size(V))));
            end

            if ~iscell(z0)
                [Z{v_ind, i_layer}, H{v_ind, i_layer}, ~] = seminmf(V, layers(i_layer), ...
                    z0, h0, bUpdateH, maxiter, tolfun, bUpdateZ, verbose, 0);
            else
                if verbose
                    display('Using existing Z');
                end
                [Z{v_ind, i_layer}, H{v_ind, i_layer}, ~] = seminmf(V, layers(i_layer), ...
                    z0{i_layer}, h0, bUpdateH, 1, tolfun, 0, verbose, 0);
            end
        end
    else
        Z = z0;
        H = h0;

        if verbose
            display('Skipping initialization, using provided init matrices...');
        end
    end
end

GG = 0;
HG = 0;
for v_ind = 1:numOfView
    GG = G{v_ind} * G{v_ind}' + GG;
    HG = H{v_ind, num_of_layers} * G{v_ind}' + HG;
end
Hc = HG * GG^(-1);

if verbose
    display('Finetuning...');
end

H_err = cell(numOfView, num_of_layers);
derror = [];
g_inv = @(x) x;

for iter = 1:20
    for v_ind = 1:numOfView
        X = XX{v_ind};
        X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 1)));

        H_err{v_ind, numel(layers)} = H{v_ind, numel(layers)};
        for i_layer = numel(layers) - 1:-1:1
            H_err{v_ind, i_layer} = Z{v_ind, i_layer + 1} * H_err{v_ind, i_layer + 1};
        end

        for i = 1:numel(layers)
            if bUpdateZ
                try
                    if i == 1
                        Z{v_ind, i} = X * pinv(H_err{v_ind, 1});
                    else
                        Z{v_ind, i} = pinv(D') * X * pinv(H_err{v_ind, i});
                    end
                catch
                    display(sprintf('Convergance error %f. min Z{i}: %f. max %f', ...
                        norm(Z{v_ind, i}, 'fro'), min(min(Z{v_ind, i})), max(max(Z{v_ind, i}))));
                end
            end

            if i == 1
                D = Z{v_ind, 1}';
            else
                D = Z{v_ind, i}' * D;
            end

            if bUpdateH && (i < numel(layers))
                A = D * X;
                Ap = (abs(A) + A) ./ 2;
                An = (abs(A) - A) ./ 2;

                B = D * D';
                Bp = (abs(B) + B) ./ 2;
                Bn = (abs(B) - B) ./ 2;

                H{v_ind, i} = H{v_ind, i} .* sqrt((Ap + Bn * H{v_ind, i}) ./ ...
                    max(An + Bp * H{v_ind, i}, 1e-10));
            end

            if (i == numel(layers)) && bUpdateLastH
                B = D * XX{v_ind} + lamda2 * Hc * G{v_ind};
                E = ones(layers(numel(layers)));
                I = eye(layers(numel(layers)));
                C = D * D' + lamda1 * E + lamda2 * I;

                Ba = (abs(B) + B) ./ 2;
                Bb = (abs(B) - B) ./ 2;
                Ca = (abs(C) + C) ./ 2;
                Cb = (abs(C) - C) ./ 2;
                A = H{v_ind, i};

                Hm_a = Ba + Cb * A;
                Hm_b = Bb + Ca * A;
                H{v_ind, i} = H{v_ind, i} .* sqrt(Hm_a ./ Hm_b);
            end
        end
    end

    GG = 0;
    HG = 0;
    for v_ind = 1:numOfView
        GG = G{v_ind} * G{v_ind}' + GG;
        HG = H{v_ind, num_of_layers} * G{v_ind}' + HG;
    end
    Hc = HG * GG^(-1);

    for v_ind = 1:numOfView
        X = XX{v_ind};
        X = bsxfun(@rdivide, X, sqrt(sum(X .^ 2, 1)));
        E = ones(layers(numel(layers)));
        dnorm(v_ind) = deep_cost(X, Z(v_ind, :), H(v_ind, :), 1, lamda1, lamda2, E, Hc, G{v_ind}, g_inv);
    end

    maxDnorm = sum(dnorm);
    derror(iter) = maxDnorm;

    if verbose
        display(sprintf('#%d error: %f', iter, maxDnorm));
    end
end

end
