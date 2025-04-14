function [x,output] = CS_L1_uncon_ADMM(A,b,pm)
%min_x .5||Ax-b||^2 + lambda |x|_1

%Input: dictionary A, data b, parameters set pm
%       pm.lambda: regularization paramter
%       pm.delta: penalty parameter for ADMM, default value: 10*lambda
%       pm.maxoit: max outer iterations, default value: 10
%       pm.maxit: max inner iterations: default value: 5000
%       pm.tol: outer tolerace, default value: 1e-3
%       pm.abstol: abs tolerance for ADMM: default value: 1e-7
%       pm.reltol: rel tolerance for ADMM: default value: 1e-5
%Output: computed coefficients x
%       output.relerr: relative error of x_n and x_{n-1}
%       output.obj: objective function of x_n:
%       obj(x) = lambda |x|_1+0.5|Ax-b|^2
%       output.res: residual of x_n: norm(Ax-b)/norm(b)
%       output.err: error to the ground-truth: norm(x-xg)/norm(xg)
%       output.relerr: error between two neighboring iterates
%       output.time: computational time

%% DCA for outer interation
[M,N] = size(A); start_time = tic; 


%     %% temp egrid
%     W = 2*ones(222,1);
%     W(34) = 0.01;


%% parameters
lambda = 1e-5; delta = 100*lambda;
maxit = 5*N; 
x0 = zeros(N,1); xg = x0;
reltol = 1e-5; %abstol = 1e-6;
eps = 1e-16;


if isfield(pm,'lambda'); lambda = pm.lambda; end
if isfield(pm,'delta'); delta = pm.delta; end
if isfield(pm,'maxit'); maxit = pm.maxit; end
if isfield(pm,'x0'); x0 = pm.x0; end
if isfield(pm,'xg'); xg = pm.xg; end
if isfield(pm,'reltol'); reltol = pm.reltol; end
if isfield(pm,'b_gt')
    b_gt  = pm.b_gt; 
    objtb = @(x) .5*norm(A*x-b_gt)^2 + lambda*(norm(x,1));
end

output.lambda = lambda;

%% pre-computing/initialize
AAt = A*A';
L = chol( speye(M) + 1/delta*AAt, 'lower' );
L = sparse(L);
U = sparse(L');

x = x0;
Atb = A'*b;
y = zeros(N,1); u = y;

obj         = @(x) .5*norm(A*x-b)^2 + lambda*(norm(x,1));

for it = 1:maxit
            
            %update x
            xold = x; 
            x =shrink(y-u, lambda/delta);
           
         
            %update y
            yold = y;
            rhs = Atb  + delta*(x+u);
            y = rhs/delta - (A'*(U\(L\(A*rhs))))/delta^2;
  
            %update u
            u = u + x-y;
    
              
 
        
    % stop conditions & outputs
    relerr = norm(x-xold)/max([norm(x),norm(xold),eps]);
%     relerr = norm(x-y)/max([norm(x),norm(y),eps]);
    residual    = norm(A*x - b);
    %% temp egrid
%     x = W.*x;
    
    %%

    
    output.relerr(it) = relerr;
    
    output.obj(it)      = obj(x);
    output.gtobj(it)    = obj(xg);
%     output.gtobjtb(it)  = objtb(xg);
    output.time(it)     = toc(start_time);
    output.res(it)      = residual;
    output.err(it)      = norm(x-xg)/norm(xg);
    output.L0(it)       = nnz(x);
    
%     if  (residual < abstol || relerr < reltol) && it > 2  
    
    if relerr < reltol && it > 2  
        break;
    end
end



end

%% Shrinkage
function z = shrink(x, r)
    z = sign(x).*max(abs(x)-r,0);
end

