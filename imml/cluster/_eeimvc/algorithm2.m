function KH2 = algorithm2(KH,S)

num = size(KH,1);
numker = size(KH,3);
KH2 = zeros(num,num,numker);

% First apply kcenter and knorm per view on observed samples
for p =1:numker
    %% missing index: S{p}.indx
    obs_indx = setdiff(1:num,S{p}.indx');
    KAp = KH(obs_indx,obs_indx,p);

    % Apply kcenter on observed kernel
    n_obs = length(obs_indx);
    D = sum(KAp) / n_obs;
    E = sum(D) / n_obs;
    J = ones(n_obs,1) * D;
    KAp = KAp - J - J' + E * ones(n_obs, n_obs);
    KAp = 0.5 * (KAp + KAp');

    % Apply knorm on observed kernel
    diag_k = diag(KAp);
    KAp = KAp ./ sqrt(diag_k * diag_k');

    % Symmetrize
    KH2(obs_indx,obs_indx,p) = (KAp+KAp')/2;
end
clear KH