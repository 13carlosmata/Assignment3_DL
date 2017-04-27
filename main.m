addpath 'cifar-10-batches-mat';
clear all;
close all;

%% Reading data and initialize the parameters of the network
fprintf('   --> Running Code \n'); 
fprintf('GD parameters '); 
GD=GDparams;
GD.n_batch=100;
GD.n_epochs=30;
GD.eta=0.0385;
lambda=0;            fprintf('- done\n'); 
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
m = [50,30]; % amount of nodes per layer
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
acc = ComputeAccuracy(trainX,trainY,W,b,L);
%%
fprintf('Computing Gradients ');
[LW,Lb,JW,Jb] = ComputeGradients(trainX, trainY, P, W,b, h, s1, lambda);    fprintf('- done \n');

