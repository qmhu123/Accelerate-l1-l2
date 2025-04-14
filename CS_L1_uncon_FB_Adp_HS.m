function [x, output] = CS_L1_uncon_FB_Adp_HS(A,b,pm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         min_x .5||Ax-b||^2 + lambda*|x|_1                           %%%  
%%%                                                                     %%%
%%% Input: dictionary A, data b, parameters set pm                      %%%
%%%        pm.lambda: regularization paramter                           %%%
%%%        pm.delta: penalty parameter for FBS                          %%%
%%%        pm.maxit: max iterations                                     %%%
%%%        pm.reltol: rel tolerance for FBS: default value: 1e-6        %%%
%%%        pm.alpha: alpha in the regularization                        %%%
%%% Output: computed coefficients x                                     %%%
%%%        output.relerr: relative error of yold and y                  %%%
%%%        output.obj: objective function of x_n:                       %%%
%%%        obj(x) = lambda(|x|_1 - alpha |x|_2)+0.5|Ax-b|^2             %%%
%%%        output.res: residual of x_n: norm(Ax-b)/norm(b)              %%%
%%%        output.err: error to the ground-truth: norm(x-xg)/norm(xg)   %%%
%%%        output.time: computational time                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,N]       = size(A); 
start_time  = tic; 


%% parameters
if isfield(pm,'lambda')
    lambda = pm.lambda; 
else
    lambda = 1e-5;  % default value
end
% parameter for ADMM
if isfield(pm,'delta'); 
    delta = pm.delta; 
else
    delta = 100 * lambda;
end
% maximum number of iterations
if isfield(pm,'maxit'); 
    maxit = pm.maxit; 
else 
    maxit = 5*N; % default value
end
% initial guess
if isfield(pm,'x0'); 
    x0 = pm.x0; 
else 
    x0 = zeros(N,1); % initial guess
end
if isfield(pm,'xg'); 
    xg = pm.xg; 
else 
    xg = x0;
end
if isfield(pm,'reltol'); 
    reltol = pm.reltol; 
else 
    reltol  = 1e-6; 
end



%% initialize
x 		= x0; 
y 		= x0;
z       = x0;
xold    = x;

% t       = 1; 
% told    = 1;


obj         = @(x) .5*norm(A*x-b)^2 + lambda*(norm(x,1));
output.pm   = pm;

f_old = 1;
s = (A'*(A*y-b));
for it = 1:maxit
    % gradient
    f_new = (A'*(A*y-b));    
    momentum_coef = min((f_new'*(f_new-f_old))/(-s'*(f_new-f_old)), 1);
    s = f_new + momentum_coef*s;
    f_old = f_new;
    
    y = x + momentum_coef*(x-xold);
    z = shrink(y - delta * A' * (A*y - b),delta*lambda);
    v = shrink(x - delta * A' * (A*x - b),delta*lambda);
     
%     told = t;
%     t = (sqrt(4*t^2+1) + 1)/2;
        
    xold = x;
    if obj(z) <= obj(v)
        x = z;
    else
        x = v;
    end
    
  
        
        
      
    % stop conditions & outputs
    relerr      = norm(x - xold)/max([norm(x), norm(xold), eps]);
    residual    = norm(A*x - b)/norm(b);
    
    output.relerr(it)   = relerr;
    output.obj(it)      = obj(x);
    output.gtobj(it)    = obj(xg);
    output.time(it)     = toc(start_time);
    output.res(it)      = residual;
    output.err(it)      = norm(x - xg)/norm(xg);
    
    if relerr < reltol && it > 2  
        break;
    elseif any(isnan(x)) && it > 2
        fprintf('blow up\n')
        break;
    end
end


end

