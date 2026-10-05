function [Y,C,WP,beta,obj] = OS_LF_IMVC_alg(KH,S,k,lambda)

num = size(KH, 2); %the number of samples
numker = size(KH, 3); %m represents the number of kernels
maxIter = 100; %the number of iterations
KH = initializeKH(KH,S);
[HP,WP] = myInitialization(KH,S,k);
HP00 = HP;
beta = ones(numker,1)*sqrt(1/numker);
[C] = myInitializationC(KH,k);
KC  = mycombFun(KH,beta);
KC = (KC+KC')/2;
H0 = mykernelkmeans(KC,k);

flag = 1;
iter = 0;

RpHpwp = zeros(num,k); % k - clusters, N - samples
for p=1:numker
    RpHpwp = RpHpwp +  beta(p)*(HP(:,:,p)*WP(:,:,p));
end
RpHpwp_lambda = RpHpwp +lambda*H0;  
while flag
    iter = iter +1;
    %---the first step-- optimize Y with given (WP C,and beta)
    YB  = RpHpwp_lambda*C';
    [Y] = mySolving(YB);
    
    %%--the second step-- optimize C with (Y, WP and beta)
    CB = Y'*RpHpwp_lambda;
    [Uh,Sh,Vh] = svd(CB,'econ');
    C = Uh*Vh';
    

    %---the third step-- optimize WP with (C, Y and beta)
    WP = updateWP_OSLFIMVC(HP,Y,C);
    
    %%--the fourth step-- optimize the Hp with given (Wp,Y,C and beta)
    for p = 1 : numker
        mis_set = S{p}.indx;
        obs_set = setdiff(1:num, mis_set);
        HB =  Y(mis_set,:) * C *WP(:,:,p)';
        [Uh,Sh,Vh] = svd(HB,'econ');
        if length(mis_set) > 0
            HP(mis_set,:,p) = Uh*Vh';
        end
        HP(obs_set,:,p) = HP00(obs_set,:,p);
    end
    
    %---the fifth step-- optimize beta with (C, Y and WP)
    beta = updateBeta_OSLFIMVC(HP,WP,Y,C);


    %---Calculate Obj--
    RpHpwp = zeros(num,k);
    for p = 1:numker
        RpHpwp = RpHpwp + beta(p)*HP(:,:,p)*WP(:,:,p);
    end
    RpHpwp_lambda = RpHpwp +lambda*H0;  
    obj(iter) = trace((Y*C)'*RpHpwp_lambda);
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-4 || iter>maxIter)
        flag =0;
    end

end
