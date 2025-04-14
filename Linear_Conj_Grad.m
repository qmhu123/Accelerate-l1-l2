% David S Watkins  - Fundamentals of matrix computations-Wiley (2010)
% Sec 8.7
% Conjugate-Gradient Algorithm for Ax=b
% Author : Mengqi Hu
% Date   : 01/22/2021

function [x,output] =  Linear_Conj_Grad(A,b,pm)
    [M,N] = size(A); start_time = tic; 
    if isfield(pm,'x0') 
        x = pm.x0; 
    else
        x = zeros(N,1);
    end
    if isfield(pm,'xg') 
        xg = pm.xg; 
    else
        xg = x;
    end
    if isfield(pm,'maxit')  
        maxit = pm.maxit; 
    else
        maxit = 5*N; 
    end    
    if isfield(pm,'reltol')
        reltol = pm.reltol;
    else
        reltol  = 1e-6;
    end
    if ~issymmetric(A)
        A0 = A'*A;
        Ab = A'*b;
    else
        A0 = A;
        Ab = b;
    end
    
    % objective function 0.5*norm(A*x-b)^2
	obj = @(t) 0.5*(A0*t-Ab)'*(A0*t-Ab);
    op = @(u) 0.5*(u-xg)'*A0*(u-xg) + Ab'*(u-xg);
    r = Ab;
    r = r-A0*x;
    p = r;
    nu = r'*r;
    for it = 1:maxit
       q = A0*p;
       mu = p'*q;
       alpha = nu/mu;
       xold = x;
       x = x + alpha*p;
       r = r - alpha*q;
       nu_p = r'*r;
       beta = nu_p/nu;
       p = r + beta * p;
       nu = nu_p;
       relerr      = norm(xold - x)/max([norm(x), norm(xold), eps]);
       residual    = norm(A*x - b)/norm(b);
       output.relerr(it)   = relerr;
       output.obj(it)      = obj(x);
       output.time(it)     = toc(start_time);
       output.res(it)      = residual;
       output.op(it)      = op(x);
       output.err(it)      = norm(x - xg)/norm(xg);
       if relerr < reltol && it > 2
           break;
       end
    end
end