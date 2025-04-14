clear; 
% close all
clc

%% parameter settings
M = 256*4; N = M*4;    % matrix dimension M-by-N
% N = 64; M = 256;    % matrix dimension M-by-N
K = 5*4 ;             % sparsity
% K = 10;             % sparsity
% K = 15;             % sparsity
rng(3000);
matrixtype = 1; 
% 1 for Gaussian
% 2 for DCT


%% construct sensing matrix
switch matrixtype 
    case 1
        A   = randn(M,N); % Gaussian matrix
        A   = A / norm(A);    
    case 2    
        A   = dctmtx(N); % dct matrix
        idx = randperm(N-1);
        A   = A([1 idx(1:M-1)+1],:); % randomly select m rows but always include row 1
        A   = A / norm(A);   
end


%% construct sparse ground-truth 

x_ref = zeros(N,1); % true vector
xs = randn(K,1);
x_ref(randsample(N,K)) = xs;

%% Given lambda, construct b, so that x is a stationary point
lambda = 1e-2;
options.dim = size(A);
% options.sparsity = 5;
options.seed = 1;
[b,x,tau,sigma,~] = construct_bpdn_instance(A,x_ref,lambda,options);

%% compare L1-L2 solvers
pm.lambda = lambda;
pm.delta = pm.lambda*100;
pm.xg = x_ref;
pm.reltol = 1e-8;
pm.b_gt = b;

pmFB = pm;
pmFB.ReSt = 20;

[xl1ADMM, outputl1ADMM] = CS_L1_uncon_ADMM(A,b,pm);
[xl1FISTA, outputl1FISTA] = CS_L1_uncon_FISTA(A,b,pm);
[xl1FB, outputl1FB] = CS_L1_uncon_FB(A,b,pmFB);
[xl1FBHB, outputl1FBHB] = CS_L1_uncon_FB(A,b,pmFB);
[xl1FB2,outputl1FBRe] = CS_L1_uncon_FB_restart(A,b,pmFB);


[xl1FBFR,outputl1FBFR] = CS_L1_uncon_FB_Adaptive(A,b,pm);
[xl1FBPR,outputl1FBPR] = CS_L1_uncon_FB_Adp_PR(A,b,pm);
[xl1FBHS,outputl1FBHS] = CS_L1_uncon_FB_Adp_HS(A,b,pm);
[xl1FBDY,outputl1FBDY] = CS_L1_uncon_FB_Adp_DY(A,b,pm);


fprintf('         FISTA & err = %.4e& t =%.4e \\\\\n', outputl1FISTA.err(end),outputl1FISTA.time(end))
fprintf('         APG & err = %.4e& t =%.4e \\\\\n', outputl1FB.err(end),outputl1FB.time(end))
fprintf('         FR & err = %.4e& t =%.4e \\\\\n', outputl1FBFR.err(end),outputl1FBFR.time(end))
fprintf('         PR & err = %.4e& t =%.4e \\\\\n', outputl1FBPR.err(end),outputl1FBPR.time(end))
fprintf('         HS & err = %.4e& t =%.4e \\\\\n', outputl1FBHS.err(end),outputl1FBHS.time(end))
fprintf('         DY & err = %.4e& t =%.4e \\\\\n', outputl1FBDY.err(end),outputl1FBDY.time(end))


% figure
% semilogy(outputl1ADMM.err,'g:', 'LineWidth',2)
% hold on
% semilogy(outputl1FISTA.err,'-r', 'LineWidth',2)
% semilogy(outputl1FB.err,'k-.', 'LineWidth',2)
% semilogy(outputl1FBHB.err,'c--', 'LineWidth',2)
% semilogy(outputl1FBRe.err,'b--', 'LineWidth',2)
% hold off


% 
% figure
% semilogy(outputl1FISTA.err,'r', 'LineWidth',2)
% hold on
% semilogy(outputl1ADMM.err,'g:', 'LineWidth',2)
% semilogy(outputl1FB.err,'k-.', 'LineWidth',2)
% semilogy(outputl1FBFR.err,'m-^', 'LineWidth',2)
% semilogy(outputl1FBPR.err,'b--', 'LineWidth',2)
% semilogy(outputl1FBHS.err,'c-x', 'LineWidth',2)
% semilogy(outputl1FBDY.err,'y-+', 'LineWidth',2)
% % semilogy(outputl1FBHB.err,'c--', 'LineWidth',2)
% % semilogy(outputl1FBRe.err,'b--', 'LineWidth',2)
% hold off
% 
% LEG1 = legend('l_1 FISTA','l_1 ADMM','l_1 FB','l_1 FR','l_1 PR','l_1 HS','l_1 DY', 'location', 'NorthEast');
% 



save(['l1Constructed_1024_Sp_',num2str(K),'.mat'],'outputl1ADMM',...
    'outputl1FISTA','outputl1FB','outputl1FBHB','outputl1FBRe',...
    'outputl1FBFR','outputl1FBPR','outputl1FBHS','outputl1FBDY','K')