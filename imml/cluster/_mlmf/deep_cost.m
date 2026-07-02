function error = deep_cost(X, Z, H, weight,lamda1,lamda2,E,Hc,G,g_inv)
%error = lamda2*norm(H{numel(H)} - Hc*G,'fro')
error = weight*norm(X - deep_recon(Z, H, g_inv), 'fro')+lamda1*trace(H{numel(H)}*H{numel(H)}'*E)+lamda2*norm(H{numel(H)} - Hc*G,'fro');
end
