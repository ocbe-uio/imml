function [H_normalized,WP,HP,obj] = IncompleteMultikernelLatefusionclusteringV1Hv(HP,k,lambda,maxIter)

num = size(HP, 1); %the number of samples
numker = size(HP, 3); %m represents the number of kernels

WP = zeros(k,k,numker);
for p =1:numker
    WP(:,:,p) = eye(k);
end
HP00 = HP;

flag = 1;
iter = 0;
% res91 = zeros(maxIter+1,3,numker);
while flag
    iter = iter +1;
    %---the first step-- optimize H_star with given (HP, WP and beta)
    RpHpwp = zeros(num,k); % k - clusters, N - samples
    for p=1:numker
        RpHpwp = RpHpwp + (HP(:,:,p)*WP(:,:,p));
    end
    [Uh,Sh,Vh] = svd(RpHpwp,'econ');
    Hstar = Uh*Vh';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     res90(iter,:) = myNMIACC(Hstar,Y,k);
    
    %---the second step-- optimize WP with (HP, H_star and beta)
    WP = updateWPabsentClusteringV1(HP,Hstar);
    
    %---the third step-- optimize HP with (WP, H_star and beta)
    HP = updateHPabsentClusteringV1(WP,Hstar,lambda,HP00);
%     for p=1:numker
%         res91(iter,:,p) = myNMIACC(HP(:,:,p),Y,k);
%     end
    
    %---Calculate Obj--
    RpHpwp = zeros(num,k);
    obj2 = 0;
    for p = 1:numker
        RpHpwp = RpHpwp + HP(:,:,p)*WP(:,:,p);
        obj2 = obj2 + trace(HP(:,:,p)'*HP00(:,:,p));
    end
    obj(iter) = trace(Hstar'*RpHpwp) + lambda*obj2;
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-4 || iter>maxIter)
        flag =0;
    end
%     if (iter>2) && (iter>maxIter)
%         flag =0;
%     end
end
H_normalized = Hstar./ repmat(sqrt(sum(Hstar.^2, 2)), 1,k);