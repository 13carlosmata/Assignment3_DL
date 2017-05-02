
addpath 'cifar-10-batches-mat';
clear all
close all
%% Reading data and initialize the parameters of the network
%0.0095 and 1e-5
fprintf('   --> Running Code \n'); 
fprintf('GD parameters '); 
GD=GDparams;
GD.n_batch=100;
GD.n_epochs=10;
GD.eta=0.0095;
lambda=1e-5;            fprintf('- done\n'); 
%%  Loading Batches from CIFAR  (for training, validation and testing
fprintf('Loading Batch '); 
[trainX, trainY, trainy] = LoadBatch('data_batch_1');              
[valX, valY, valy] = LoadBatch('data_batch_2');                    
[testX, testY, testy] = LoadBatch('test_batch.mat');               fprintf('- done \n');  
fprintf('Preprocessing data '); 

mean_trainX = mean(trainX, 2);
trainX = trainX - repmat(mean_trainX, [1, size(trainX, 2)]);
fprintf('- done\n');      fprintf('Running substractions with mean_X '); 

valX = valX - repmat(mean_trainX, [1, size(valX, 2)]);
testX = testX - repmat(mean_trainX, [1, size(testX, 2)]);
fprintf('- done\n');
%% Initiating parameters
fprintf('Initialization of W{} and b{} ');
L=3; % amount of layers
% m = repmat(50,1,L-1); % amount of nodes per layer
m = [50,30];
K = 10; %labels
d=size(trainX,1); % amount of images 
[W,b] = InitParams(d,m,K,L); fprintf('- done \n');
%% 
fprintf('Evaluating Classifier ');
[P,h,s] = EvaluateClassifier(trainX, W, b, L);        fprintf('- done \n');
%%
fprintf('Computing Cost ');
[J] = ComputeCost(trainX,trainY,W,b,lambda,L);     fprintf('- done \n');
fprintf('      > Initial Loss = %f\n', J);
%%
fprintf('Computing Accuracy ');
acc = ComputeAccuracy(trainX,trainY,W,b,L);         fprintf('- done \n');
fprintf('      > Initial ACC = %f\n',acc);
%%
fprintf('Computing Gradients ');
[grad_W,grad_b] = ComputeGrad3(trainX,trainY,W,b,P,h,s,lambda,L);  fprintf('- done \n');
%%
% fprintf('Evaluating the difference between the numerical and analytical gradients: \n');
% % Numerical vs analitical
% [P_grad,h_grad,s_grad] = EvaluateClassifier(trainX(:,1), W, b, L);
% fprintf('    *Analitical');
% [grad_W,grad_b] = ComputeGrad3(trainX(:,1),trainY(:,1),W,b,P_grad,h_grad,s_grad,lambda,L);
% fprintf(' -> calculated \n');
% fprintf('    *Numerical');
% hexr = 1e-9;
% [grad_b_num, grad_W_num] = ComputeGradsNum(trainX(:,1), trainY(:,1), W,b, lambda, hexr,L);
% fprintf(' -> calculated \n');
% 
% fprintf('\n Numerical-Analitical comparison done\n');
% for wi=1:L
%     diff_W(wi)=abs(max(max(grad_W{wi}-grad_W_num{wi})));
%     fprintf(['   Worst approximation for W',num2str(wi),': ',num2str(diff_W(wi)),'\n']);        
% end
% for bi=1:L
%     diff_b(bi)=abs(max(max(grad_b{bi}-grad_b_num{bi})));
%     fprintf(['   Worst approximation for b',num2str(bi),': ',num2str(diff_b(bi)),'\n']);
% end
% fprintf(' \n');
%% 
fprintf('Unorm Minibatch ');
[Wstar,bstar,JK] = MiniBatchGD(trainX, trainY, GD, W,b, lambda, L); fprintf('done \n');
acc_New = ComputeAccuracy(trainX,trainY,Wstar,bstar,L); 
fprintf('Norm Minibatch ');
[Wnorm,bnorm,JK_norm,u,v] = BatchNorm(trainX, trainY, GD, W,b, lambda, L); fprintf('- done \n');
acc_norm = ComputeAccuracy(trainX,trainY,Wnorm,bnorm,L); 
fprintf('      > ACC for Training data = %f\n',acc_norm);
figure
plot(0:GD.n_epochs,JK,0:GD.n_epochs,JK_norm);
legend('Unnormalized Batch','Normalized Batch');
title('Comparison with and without normalization')
xlabel('Using 50 and 30 nodes for 1st and 2nd hidden layers respectively');

fprintf('--> Checking Validated data \n');
fprintf('Unnormalized batch');
[Wstar_val_u,bstar_val_u,JK_val_u] = MiniBatchGD(valX, valY, GD, W,b, lambda, L); fprintf(' - done \n');
acc_val_u = ComputeAccuracy(trainX,trainY,Wstar,bstar,L); 
fprintf('Normalized batch');
[Wstar_val_n,bstar_val_n,JK_val_n] = BatchNorm(valX, valY, GD, W,b, lambda, L); fprintf(' - done \n');
acc_val_n = ComputeAccuracy(valX,valY,Wnorm,bnorm,L); 
fprintf(' > ACC for validated data (Unnormalized batch) = %f\n',acc_val_u);
fprintf(' > ACC for validated data (Normalized batch) = %f\n',acc_val_n);
acc_test = ComputeAccuracy(testX,testY,Wnorm,bnorm,L); 
fprintf(' > ACC for Testing data (Normalized)= %f\n',acc_test);
% acc_test_u = ComputeAccuracy(testX,testY,Wstar,bstar,L); 
% fprintf('      > ACC for Testing data (Without Normalization) = %f\n',acc_test_u);
%%
fig_i=figure;
xplot=0:GD.n_epochs;
plot(xplot,JK,xplot,JK_norm,xplot,JK_val_u,xplot,JK_val_n);
legend(['Training Loss - UNnorm','   ',num2str(acc_New),'%'],...
    ['Training Loss - Norm','   ',num2str(acc_norm),'%'],...
    ['Validation Loss - UNnorm','   ',num2str(acc_val_u),'%']);
title(['Parameters used: ', ' n.batch: ',num2str(GD.n_batch),' epochs: ',num2str(GD.n_epochs),' eta: ',num2str(GD.eta),' lambda: ',num2str(lambda), ' k-layers: ',num2str(L)],'FontSize',12);
xlabel(['Test accuracy (Normalized): ',num2str(acc_test),'%']);
%saveas(fig_i,['C:\Users\cmata_oloq6sf\Dropbox\Medical Engineering\II Semestre\Deep Learning\Assignment3_DL\fig\','h',num2str(hour(datetime)),'m',num2str(minute(datetime)),'s',num2str(second(datetime),2),'.jpg']);
% close (fig_i)
fprintf('\n > Code ran succesfully < \n')

