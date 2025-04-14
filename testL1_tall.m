clear;
close all
clc

%% parameter settings
N = 256; M = N*4;    % matrix dimension M-by-N
% K = 1 ;             % sparsity
% K = 3 ;             % sparsity
K = 5 ;             % sparsity
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

b_gt = A*x_ref;
SNR = 30;
SNR12 = 10^(SNR/10);
N = length(b_gt);
% Standard Deviation of noise
noise_Var = (sum(b_gt.^2) / SNR12 / N );
noise_std = sqrt(noise_Var);
noise_real = noise_std * randn(N,1);

Snr = snr(b_gt,noise_real);
b = b_gt + noise_real;
% b = b_gt;
pm.b_gt = b_gt;
pm.xg = x_ref;
pm.reltol = 1e-8;

%% parameters tuning
% 
% lambdatest = 5;
% err = zeros(lambdatest,1);
% objerr = err;
% cycle = err;
% disp(['Objective  Ground truth'])
% for  lambdatrial = 1:lambdatest
% %     pm.lambda = 10^(lambdatrial-12);
% % %     1e-3
% %     pm.lambda = 1e-3*lambdatrial;
% %     pm.lambda = 2.5e-3+(lambdatrial-3)*1*0.1e-3;
%     pm.lambda =  2.5e-3;
% %     disp(.5*norm(A*x_ref-b)^2+ pm.lambda*(norm(x_ref,1)))
% %     pm.delta = pm.lambda*1e2;
% %     pm.delta = pm.lambda*10^(lambdatrial-2);
% %     pm.delta = pm.lambda*2e2*lambdatrial;
%     pm.delta = pm.lambda*(7e2 + 0.1e2*(lambdatrial-3));
%     
% %     pm.ReSt = 10;
%     
% %%  testing parameter
% 
% %     [xl1ADMM, outputl1] = CS_L1_uncon_ADMM(A,b,pm);
% %     [xl1FISTA, outputl1] = CS_L1_uncon_FISTA(A,b,pm);
%     [xl1FB, outputl1] = CS_L1_uncon_FB(A,b,pm);
% %     [xl1FBHB, outputl1] = CS_L1_uncon_FB(A,b,pm);
% %     [xl1FBAdp, outputl1] = CS_L1_uncon_FB_Adaptive(A,b,pm);
% %     [xl1FB2,outputl1] = CS_L1_uncon_FB_restart(A,b,pm);
% %     [xl1FBPR,outputl1] = CS_L1_uncon_FB_Adp_PR(A,b,pm);
% %     [xl1FBHS,outputl1] = CS_L1_uncon_FB_Adp_HS(A,b,pm);
% %     [xl1FBDY,outputl1] = CS_L1_uncon_FB_Adp_DY(A,b,pm);
% %     
% %     disp(pm.lambda)
% %     disp([(outputl1.obj(end)>= outputl1.gtobj(end)),outputl1.err(end),length(outputl1.err)<1280])
%     objerr(lambdatrial) = (outputl1.obj(end)-outputl1.gtobj(end));
%     err(lambdatrial) = outputl1.err(end);
%     cycle(lambdatrial) = length(outputl1.err);
%     
% %     figure
% %     semilogy(outputl1.obj)
% %     hold on 
% %     semilogy(outputl1.gtobj)
% end
% 
% 
% fprintf('\tErr \t Objerr\t cycle\n')
% disp([err,objerr,cycle/1e3])


%% 30dB

pm.lambda =  2.5e-3;
pmADMM = pm;
pmADMM.delta = pmADMM.lambda*1.9e2;
pmFISTA = pm;
pmFISTA.delta = pmFISTA.lambda*6.1e2;
pmFB = pm;
pmFB.delta = pmFB.lambda*7e2;
pmFBHB = pm;
pmFBHB.delta = pmFB.lambda*6.0e2;
pmFBARe = pm;
pmFBARe.delta = pmFBARe.lambda*7.4e2;
pmFBARe.ReSt = 10;





pmFBFR = pm;
pmFBFR.delta = pmFBFR.lambda*5.5e2;
pmFBPR = pm;
pmFBPR.delta = pmFBFR.lambda*6.8e2;
pmFBHS = pm;
pmFBHS.delta = pmFBHS.lambda*6.0e2;
pmFBDY = pm;
pmFBDY.delta = pmFBDY.lambda*7.4e2;


%% no noise
% 
% pm.lambda =  2.5e-3;
% pmADMM = pm;
% pmADMM.delta = pmADMM.lambda*1.9e2;
% pmFISTA = pm;
% pmFISTA.delta = pmFISTA.lambda*6.1e2;
% pmFB = pm;
% pmFB.delta = pmFB.lambda*7.5e2;
% pmFBHB = pm;
% pmFBHB.delta = pmFB.lambda*6.0e2;
% pmFBAdp = pm;
% pmFBAdp.delta = pmFBAdp.lambda*5.5e2;
% pmFBARe = pm;
% pmFBARe.delta = pmFBARe.lambda*7.4e2;
% pmFBARe.ReSt = 10;


%% for actual calculation

[xl1ADMM, outputl1ADMM] = CS_L1_uncon_ADMM(A,b,pmADMM);
[xl1FISTA, outputl1FISTA] = CS_L1_uncon_FISTA(A,b,pmFISTA);
[xl1FB, outputl1FB] = CS_L1_uncon_FB(A,b,pmFB);
[xl1FBHB, outputl1FBHB] = CS_L1_uncon_FB(A,b,pmFBHB);
[xl1FB2,outputl1FBRe] = CS_L1_uncon_FB_restart(A,b,pmFBARe);


[xl1FBFR,outputl1FBFR] = CS_L1_uncon_FB_Adaptive(A,b,pmFBFR);
[xl1FBPR,outputl1FBPR] = CS_L1_uncon_FB_Adp_PR(A,b,pmFBPR);
[xl1FBHS,outputl1FBHS] = CS_L1_uncon_FB_Adp_HS(A,b,pmFBHS);
[xl1FBDY,outputl1FBDY] = CS_L1_uncon_FB_Adp_DY(A,b,pmFBDY);

% disp([length(outputl1ADMM.err),length(outputl1FISTA.err),length(outputl1FB.err),length(outputl1FBHB.err),...
%     length(outputl1FBAdp.err),length(outputl1FBRe.err)])

% figure
% semilogy(outputl1FISTA.time,outputl1FISTA.err,'r', 'LineWidth',2)
% hold on
% semilogy(outputl1ADMM.time,outputl1ADMM.err,'g:', 'LineWidth',2)
% semilogy(outputl1FB.time,outputl1FB.err,'k-.', 'LineWidth',2)
% semilogy(outputl1FBHB.time,outputl1FB.err,'p.', 'LineWidth',2)
% semilogy(outputl1FBAdp.time,outputl1FBAdp.err,'m--', 'LineWidth',2)
% semilogy(outputl1FBRe.time,outputl1FBRe.err,'b--', 'LineWidth',2)
% hold off
% 
% 
figure
semilogy(outputl1FISTA.err,'r', 'LineWidth',2)
hold on
semilogy(outputl1ADMM.err,'g:', 'LineWidth',2)
semilogy(outputl1FB.err,'k-.', 'LineWidth',2)
semilogy(outputl1FBHB.err,'c--', 'LineWidth',2)
semilogy(outputl1FBFR.err,'m-^', 'LineWidth',2)
semilogy(outputl1FBPR.err,'b--', 'LineWidth',2)
hold off
% 
% % % % 
% % ylim([5e-5,1])
% % LEG1 = legend('l_1 FISTA','l_1 ADMM','l_1 FB','l_1 FBHB','l_1 FBAdaptive','l_1 FBSrestart', 'location', 'NorthEast');
% LEG1 = legend('l_1 FISTA','l_1 FB','l_1 FBAdaptive','l_1 FBPR', 'location', 'NorthEast');
save(['l1data_Sp_',num2str(K),'same_l_tall','.mat'],'outputl1FISTA','outputl1ADMM','outputl1FB','outputl1FBHB',...
    'outputl1FBRe','outputl1FBFR','outputl1FBPR','outputl1FBHS','outputl1FBDY','Snr','K')