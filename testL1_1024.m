% clear;
clc
close all

%% parameter settings
M = 256; M=M*2;
N = M * 4;    % matrix dimension M-by-N
% K = 1 ;             % sparsity
% K = 3 ;             % sparsity
% K = 5 ;             % sparsity
% K = 10;             % sparsity
K = 20;             % sparsity
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

b = b_gt + noise_real;
Snr = snr(b_gt,noise_real);

% b = b_gt;
pm.b_gt = b_gt;
pm.xg = x_ref;
pm.reltol = 1e-8;
% pm.reltol = 1e-4;

%% parameters tuning
% % lambdatest = 5;
% % err = zeros(lambdatest,1);
% % objerr = err;
% % cycle = err;
% % time = err;
% % 
% % fprintf('\tErr    \tObjerr \tTime   \tCycle\n')
% % for  lambdatrial = 1:lambdatest
% % %     pm.lambda = 10^(lambdatrial-5);
% % %     pm.lambda = 0.5e-3*lambdatrial;
% % %     pm.lambda = 1.2e-3+(lambdatrial-3)*0.1e-3;
% % %     pm.lambda =  1.2e-3;
% % 
% % %     pm.delta = 1e-2;
% % %     pm.delta = pm.lambda*1e2;
% % %     pm.delta = pm.lambda*10^(lambdatrial-1);
% % %     pm.delta = pm.lambda*5e2*lambdatrial;
% %     pm.delta = pm.lambda*(12e2 + 1e2*(lambdatrial-3));
% % % %     
% % %     pm.ReSt = 20;
% %     
% %     %for testing parameter
% % %     [xl1ADMM, outputl1] = CS_L1_uncon_ADMM(A,b,pm);
% % %     [xl1FISTA, outputl1] = CS_L1_uncon_FISTA(A,b,pm);
% % %     [xl1FB, outputl1] = CS_L1_uncon_FB(A,b,pm);
% % %     [xl1FB, outputl1] = CS_L1_uncon_FB_HB(A,b,pm);
% %     [xl1FBFR, outputl1] = CS_L1_uncon_FB_Adaptive(A,b,pm);
% % %     [xl1FB2,outputl1] = CS_L1_uncon_FB_restart(A,b,pm);
% % %     [xl1FBPR,outputl1] = CS_L1_uncon_FB_Adp_PR(A,b,pm);
% % %     [xl1FBHS,outputl1] = CS_L1_uncon_FB_Adp_HS(A,b,pm);
% % %     [xl1FBDY,outputl1] = CS_L1_uncon_FB_Adp_DY(A,b,pm);
% %     
% % %     disp(pm.lambda)
% % %     disp([(outputl1.obj(end)>= outputl1.gtobj(end)),outputl1.err(end),length(outputl1.err)<1280])
% %     objerr(lambdatrial) = (outputl1.obj(end)-outputl1.gtobj(end));
% %     err(lambdatrial) = outputl1.err(end);
% %     time(lambdatrial)  =  outputl1.time(end);
% %     cycle(lambdatrial) = length(outputl1.err);
% % 
% % 
% %     fprintf('\t%.4e\t%.4e\t%.4e\t%.4e \n',err(lambdatrial),objerr(lambdatrial),time(lambdatrial),cycle(lambdatrial))
% %     
% % %     figure
% % %     semilogy(outputl1.obj)
% % %     hold on 
% % %     semilogy(outputl1.gtobj)
% % end

% 

%% 30dB
pm.lambda =  1.2e-3;

pmADMM = pm;
pmADMM.delta = pmADMM.lambda*1.1e2;

pmFISTA = pm;
pmFISTA.delta = pmFISTA.lambda*1.3e3;

pmFB = pm;
pmFB.delta = pmFB.lambda*1.5e3;

pmFBHB = pm;
pmFBHB.delta = pmFBHB.lambda*1.2e3;

pmFBARe = pm;
pmFBARe.delta = pmFBARe.lambda*1.6e3;
pmFBARe.ReSt = 10;


pmFBFR = pm;
pmFBFR.delta = pmFBFR.lambda*1.3e3;
pmFBPR = pm;
pmFBPR.delta = pmFBFR.lambda*1.5e3;
pmFBHS = pm;
pmFBHS.delta = pmFBHS.lambda*1.6e3;
pmFBDY = pm;
pmFBDY.delta = pmFBDY.lambda*1.5e3;



%% for actual calculation


[xl1ADMM, outputl1ADMM] = CS_L1_uncon_ADMM(A,b,pmADMM);
fprintf('         ADMM & err = %.4e& t =%.4e \\\\\n', outputl1ADMM.err(end),outputl1ADMM.time(end))
% fprintf('         ADMM & %.4e&%.4e \\\\\n', outputl1ADMM.err(end),outputl1ADMM.time(end))

[xl1FISTA, outputl1FISTA] = CS_L1_uncon_FISTA(A,b,pmFISTA);
fprintf('         FISTA & err = %.4e& t =%.4e \\\\\n', outputl1FISTA.err(end),outputl1FISTA.time(end))
% fprintf('         FISTA & %.4e&%.4e \\\\\n', outputl1FISTA.err(end),outputl1FISTA.time(end))

[xl1FB, outputl1FB] = CS_L1_uncon_FB(A,b,pmFB);
fprintf('         APG & err = %.4e& t =%.4e \\\\\n', outputl1FB.err(end),outputl1FB.time(end))
% fprintf('         outputl1FB & %.4e&%.4e \\\\\n', outputl1FB.err(end),outputl1FB.time(end))

[xl1FBHB, outputl1FBHB] = CS_L1_uncon_FB(A,b,pmFBHB);
fprintf('         outputl1FBHB & err = %.4e& t =%.4e \\\\\n', outputl1FBHB.err(end),outputl1FBHB.time(end))
% fprintf('         outputl1FBHB & %.4e&%.4e \\\\\n', outputl1FBHB.err(end),outputl1FBHB.time(end))

[xl1FBFR,outputl1FBFR] = CS_L1_uncon_FB_Adaptive(A,b,pmFBFR);
fprintf('         FBFR & err = %.4e& t =%.4e \\\\\n', outputl1FBFR.err(end),outputl1FBFR.time(end))
% fprintf('         outputl1FBFR & %.4e&%.4e \\\\\n', outputl1FBFR.err(end),outputl1FBFR.time(end))

[xl1FBPR,outputl1FBPR] = CS_L1_uncon_FB_Adp_PR(A,b,pmFBPR);
fprintf('         FBPR & err = %.4e& t =%.4e \\\\\n', outputl1FBPR.err(end),outputl1FBPR.time(end))
% fprintf('         outputl1FBPR & %.4e&%.4e \\\\\n', outputl1FBPR.err(end),outputl1FBPR.time(end))

[xl1FBHS,outputl1FBHS] = CS_L1_uncon_FB_Adp_HS(A,b,pmFBHS);
fprintf('         FBHS & err = %.4e& t =%.4e \\\\\n', outputl1FBHS.err(end),outputl1FBHS.time(end))
% fprintf('         outputl1FBHS & %.4e&%.4e \\\\\n', outputl1FBHS.err(end),outputl1FBHS.time(end))

[xl1FBDY,outputl1FBDY] = CS_L1_uncon_FB_Adp_DY(A,b,pmFBDY);
fprintf('         FBDY & err = %.4e& t =%.4e \\\\\n', outputl1FBDY.err(end),outputl1FBDY.time(end))
% fprintf('         outputl1FBDY & %.4e&%.4e \\\\\n', outputl1FBDY.err(end),outputl1FBDY.time(end))
% 
[xl1FB2,outputl1FBRe] = CS_L1_uncon_FB_restart(A,b,pmFBARe);
fprintf('         FBRe & err = %.4e& t =%.4e \\\\\n', outputl1FBRe.err(end),outputl1FBRe.time(end))
% fprintf('         outputl1FBRe & %.4e&%.4e \\\\\n', outputl1FBRe.err(end),outputl1FBRe.time(end))
 




% figure
% semilogy(outputl1FISTA.err,'r', 'LineWidth',2)
% hold on
% semilogy(outputl1ADMM.err,'g:', 'LineWidth',2)
% semilogy(outputl1FB.err,'k-.', 'LineWidth',2)
% semilogy(outputl1FBFR.err,'m-^', 'LineWidth',2)
% semilogy(outputl1FBPR.err,'b--', 'LineWidth',2)
% semilogy(outputl1FBHS.err,'c-x', 'LineWidth',2)
% semilogy(outputl1FBDY.err,'y-+', 'LineWidth',2)
% semilogy(outputl1FBHB.err,'c--', 'LineWidth',2)
% semilogy(outputl1FBRe.err,'b--', 'LineWidth',2)
% hold off

% LEG1 = legend('l_1 FISTA','l_1 ADMM','l_1 FB','l_1 FR','l_1 PR','l_1 HS','l_1 DY', 'location', 'NorthEast');
% save(['l1_',num2str(M),'_data_Sp_',num2str(K),'_Snr_',num2str(SNR),'same_l','.mat'],'outputl1FISTA','outputl1ADMM','outputl1FB',...
%     'outputl1FBHB','outputl1FBFR','outputl1FBPR','outputl1FBHS','outputl1FBDY','Snr','K')