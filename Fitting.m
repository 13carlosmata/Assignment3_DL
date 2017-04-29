function [params] = Fitting(lambda_i,eta_i,L_i)
addpath 'cifar-10-batches-mat';
%% Reading data and initialize the parameters of the network
fprintf('   --> Running Code \n'); 
fprintf('GD parameters '); 
GD=GDparams;
GD.n_batch=100;
GD.n_epochs=10;
GD.eta=eta_i;
lambda=lambda_i;            fprintf('- done\n'); 
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
L=L_i; % amount of layers
% m = repmat(50,1,L-1); % amount of nodes per layer
m = [50,50];
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
% 
%% Numerical vs analitical
% [P,h,s] = EvaluateClassifier(trainX(:,1), W, b, L);
% [grad_W,grad_b] = ComputeGrad3(trainX(:,1),trainY(:,1),W,b,P,h,s,lambda,L);
% fprintf('    *Analitical calculated \n');
% hexr = 1e-5;
% [grad_b, grad_W] = ComputeGradsNum(trainX(:,1), trainY(:,1), W,b, lambda, hexr,L);
% fprintf('    *Numerical calculated \n');
% beep;
%% 
fprintf('Unorm Minibatch ');
[Wstar,bstar,JK] = MiniBatchGD(trainX, trainY, GD, W,b, lambda, L); fprintf('done \n');
acc_New = ComputeAccuracy(trainX,trainY,Wstar,bstar,L); 
fprintf('Norm Minibatch ');
[Wnorm,bnorm,JK_norm,u,v] = BatchNorm(trainX, trainY, GD, W,b, lambda, L); fprintf('- done \n');
acc_norm = ComputeAccuracy(trainX,trainY,Wnorm,bnorm,L); 
plot(0:GD.n_epochs,JK,0:GD.n_epochs,JK_norm);
legend(['Unnormalized Batch','     ',num2str(acc_New),'%'],...
      ['Normalized Batch','   ',num2str(acc_norm),'%']);
%
fprintf('Checking Validated data ');
[Wstar_val,bstar_val,JK_val] = MiniBatchGD(valX, valY, GD, W,b, lambda, L); fprintf('done \n');

% %%
% acc_New = ComputeAccuracy(trainX,trainY,Wstar,bstar,L); 
% fprintf('      > New ACC = %f\n',acc_New);
% acc_New_val = ComputeAccuracy(valX,valY,Wstar,bstar,L); 
% fprintf('      > New ACC = %f\n',acc_New_val);
%%

%plot(0:GD.n_epochs,JK,0:GD.n_epochs,JK_norm);
% legend(['training loss','     ',num2str(acc_New),'%'],...
%     ['validation loss','   ',num2str(acc_New_val),'%']);
% title(['Parameters used: ', ' n.batch: ',num2str(GD.n_batch),' epochs: ',num2str(GD.n_epochs),' eta: ',num2str(GD.eta),' lambda: ',num2str(lambda), ' k-layers: ',num2str(L)],'FontSize',15);
% 
fprintf('\n Code ran succesfully \n')
params={lambda,GD.eta,L,JK_norm,acc_norm};
