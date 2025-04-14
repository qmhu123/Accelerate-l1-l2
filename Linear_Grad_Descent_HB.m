% Generic Descent Algorithm with Nesterov for Ax=b
% Author : Mengqi Hu
% Date   : 01/22/2021

function [y,output] =  Linear_Grad_Descent_HB(A,b,pm)
    [M,N] = size(A); 
    start_time = tic; 
    if isfield(pm,'x0') 
        y = pm.x0; 
    else
        y = zeros(N,1);
    end
    if isfield(pm,'xg') 
        xg = pm.xg; 
    else
        xg = y;
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
    if isfield(pm,'alpha')  
        alpha = pm.alpha; 
    else
        alpha = 1e-3;
    end    
    if isfield(pm,'beta')
        beta = pm.beta;
    else
        beta  = 1;
    end
    if ~issymmetric(A)
        A0 = A'*A;
        Ab = A'*b;
    else
        A0 = A;
        Ab = b;
    end
    
    % objective function 0.5*norm(A*x-b)^2
% 	obj = @(t) 0.5*(A0*t-Ab)'*(A0*t-Ab);
    obj = @(u) 0.5*(u)'*A0*(u) - Ab'*(u);
    r = Ab;
    r = r-A0*y;
    p = r;
    x = y;
    
    for it = 1:maxit
        
       q = A0*p;  % comment for fixed
       alpha = (p'*r)/(p'*q);

       xold = x;
       x = y + alpha*p;
       
       yold = y;
       y = x + beta*(x-xold);
       
       r = Ab - A0 *y;
       p = r;
       
       relerr      = norm(yold - y)/max([norm(y), norm(yold), eps]);
       residual    = norm(A*y - b)/norm(b);
       output.relerr(it)   = relerr;
       output.obj(it)      = obj(y);
       output.objd(it)     = abs(obj(y) - obj(xg));
%        output.op(it)      = op(x);
       output.time(it)     = toc(start_time);
       output.res(it)      = residual;
       output.err(it)      = norm(y - xg)/norm(xg);
       if relerr < reltol && it > 2
           break;
       end
    end
    output.r = r;
end 