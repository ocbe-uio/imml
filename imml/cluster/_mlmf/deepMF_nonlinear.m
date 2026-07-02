function [Hc, H, derror, Z] = deepMF_nonlinear(XX, layers, lamda1, lamda2, z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, nonlinearity)
% Nonlinear MLMF, adapted from the original MATLAB implementation for
% Octave execution through oct2py.

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

g = @(x) x;
g_inv = @(x) x;
g_inv_diff = @(x) x;

if strcmp(nonlinearity, 'tanh') == 1
    g = @(x) 3 .* atanh(x ./ 1.7159) ./ 2;
    g_inv = @(x) 1.7159 * tanh(2 / 3 * x);
    g_inv_diff = @(x) 1.7159 * 2 / 3 .* (sech((2 .* x) ./ 3) .^ 2);
elseif strcmp(nonlinearity, 'square') == 1
    g = @(x) x .^ 0.5;
    g_inv = @(x) x .* x;
    g_inv_diff = @(x) 2 * x;
elseif strcmp(nonlinearity, 'sigmoid') == 1
    sigmoid = @(x) (1 ./ (1 + exp(-x)));
    g_inv = sigmoid;
    g_inv_diff = @(x) sigmoid(x) .* (1 - sigmoid(x));
    g = @(x) log(x ./ (1 - x));
elseif strcmp(nonlinearity, 'softplus') == 1
    g_inv = @(x) log(1 + exp(x));
    g_inv_diff = @(x) exp(x) ./ (1 + exp(x));
    g = @(x) log(exp(x) - 1);
end

for v_ind = 1:numOfView
    X = XX{v_ind};

    if ~iscell(z0) && ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                V = X;
            else
                V = g(H{v_ind, i_layer - 1});
            end

            if verbose
                display(sprintf('Initialising Layer #%d...', i_layer));
            end

            [Z{v_ind, i_layer}, H{v_ind, i_layer}, ~] = seminmf(V, layers(i_layer), ...
                z0, h0, bUpdateH, maxiter, tolfun, 1, verbose, 0);
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

for iter = 1:30
    for v_ind = 1:numOfView
        X = XX{v_ind};
        E = ones(layers(numel(layers)));

        dnorm(v_ind) = deep_cost(X, Z(v_ind, :), H(v_ind, :), 1, lamda1, lamda2, E, Hc, G{v_ind}, g_inv);

        for i = numel(layers):-1:1
            if i == 2
                KSI = Z{v_ind, 1}' * X;
                PSI = Z{v_ind, 1}' * Z{v_ind, 1};
            end

            if bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))
                if i == 1
                    H{v_ind, 1} = g_inv(Z{v_ind, 2} * H{v_ind, 2});
                    H{v_ind, i}(H{v_ind, i} <= 0) = eps;
                else
                    c = g_inv_diff(Z{v_ind, 2} * H{v_ind, 2});

                    A = 2 * KSI;
                    B = 2 * PSI * g_inv(Z{v_ind, 2} * H{v_ind, 2});
                    E = ones(layers(numel(layers)));
                    F = 2 * lamda1 * E * H{v_ind, i};
                    P = 2 * lamda2 * (H{v_ind, i} - Hc * G{v_ind});

                    C = Z{v_ind, 2}' * ((B - A) .* c) + F + P;
                    [H(v_ind, :), ~] = gd_H(X, Z(v_ind, :), H(v_ind, :), C, i, g_inv, ...
                        dnorm(v_ind), lamda1, lamda2, E, Hc, G{v_ind}, verbose);
                end
            end

            dnorm(v_ind) = deep_cost(X, Z(v_ind, :), H(v_ind, :), 1, lamda1, lamda2, E, Hc, G{v_ind}, g_inv);

            if i == 1
                Z{v_ind, i} = X * pinv(g_inv(Z{v_ind, 2} * H{v_ind, 2}));
            else
                c = g_inv_diff(Z{v_ind, 2} * H{v_ind, 2});
                C = ((Z{v_ind, 1}' * (Z{v_ind, 1} * g_inv(Z{v_ind, 2} * H{v_ind, 2}) - X) .* c)) * H{v_ind, 2}';
                [Z(v_ind, :), ~] = gd_Z(X, Z(v_ind, :), H(v_ind, :), C, i, g_inv, ...
                    dnorm(v_ind), lamda1, lamda2, E, Hc, G{v_ind}, verbose);
            end

            dnorm(v_ind) = deep_cost(X, Z(v_ind, :), H(v_ind, :), 1, lamda1, lamda2, E, Hc, G{v_ind}, g_inv);
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
        E = ones(layers(numel(layers)));
        norm_(v_ind) = deep_cost(X, Z(v_ind, :), H(v_ind, :), 1, lamda1, lamda2, E, Hc, G{v_ind}, g_inv);
    end

    maxDnorm = sum(norm_);
    derror(iter) = maxDnorm;

    if verbose
        display(sprintf('#%d error: %f', iter, maxDnorm));
    end
end

end

function [Z, dnorm1] = gd_Z(X, Z, H, c, i, g_inv, dnorm, lamda1, lamda2, E, Hc, G, verbose)
eta = 0.01;
oldZ = Z{i};

while 1
    eta = eta / 2;
    Z{i} = oldZ - eta .* c;

    dnorm1 = deep_cost(X, Z, H, 1, lamda1, lamda2, E, Hc, G, g_inv);

    if eta < 0.00001
        Z{i} = oldZ;
        dnorm1 = dnorm;
        break;
    end

    if dnorm1 <= dnorm
        if verbose
            fprintf(1, 'Z(%d) eta: %f dnorm: %f\n', i, eta, dnorm1);
        end
        break;
    end
end
end

function [H, dnorm1] = gd_H(X, Z, H, c, i, g_inv, dnorm, lamda1, lamda2, E, Hc, G, verbose)
eta = 0.01;
oldH = H{i};

if i == 1
    dnorm = norm(X - Z{1} * H{1}, 'fro');
end

while 1
    eta = eta / 2;
    H{i} = oldH - eta .* c;
    H{i}(H{i} <= 0) = eps;

    if i == 1
        dnorm1 = norm(X - Z{1} * H{1}, 'fro');
    else
        dnorm1 = deep_cost(X, Z, H, 1, lamda1, lamda2, E, Hc, G, g_inv);
    end

    if eta < 0.00001
        H{i} = oldH;
        dnorm1 = dnorm;
        break;
    end

    if dnorm1 <= dnorm
        if verbose
            fprintf(1, 'H(%d) eta: %f dnorm: %f\n', i, eta, dnorm1);
        end
        break;
    end
end
end
