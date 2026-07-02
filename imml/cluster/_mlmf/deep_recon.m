function [out]= deep_recon (Z,H,g_inv)

out = H{numel(H)};

for k = numel(H) : -1 : 1;
    out =  g_inv(Z{k} * out);
end
end
