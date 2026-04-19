function [GC ] = hTrust(G, S, lambda, K, maxIter)
beta = 0.01;
alpha = 0.01;

%% construct L
d = sum(S,2);
D = diag(d);
L = D - S;

[n,n] = size(G);

U = rand(n,K);
V = rand(K,K);

iter = 0;


while(iter < maxIter)
    
UU = U'*U;

A = U*(V'*(UU*V)) + U*(V*(UU*V')) + alpha*U + lambda*D*U + 1e-8;
B = G'*U*V + G*U*V' + lambda*S*U;

U = U .* sqrt(B./A);

UU = U'*U;
AV = UU * V * UU + beta*V + 1e-8;    
BV = U'*G*U;

V = V .* sqrt(BV./AV);

Obj  = norm( (G-U*V*U'),'fro')^2 + alpha* norm(U,'fro')^2 + beta*norm(V,'fro')^2 + lambda*trace(U'*L*U);
sprintf('the object value in iter %d is %f', iter, Obj)

iter = iter + 1;
end

GC = U*V*U';

end

