function [Wstar,bstar,JK] = MiniBatchGD(X, Y, GD, W,b, lambda)
N=size(X,2);
[J,J1] = ComputeCost(X,Y,W,b,lambda);
JK=[J];
decay_rate=0.95;
v_W{1}=zeros(size(W{1})); v_W{2}=zeros(size(W{2}));
v_b{1}=zeros(size(b{1})); v_b{2}=zeros(size(b{2}));
wb = waitbar(0,'1','Name','Minibatch progress');
rho=0.9;
et=GD.eta;
for i=1:GD.n_epochs
%      fprintf('i = %d\n', i);
    waitbar(i/GD.n_epochs,wb,strcat('Epoch: ',num2str(i)));
    for j=1:N/GD.n_batch
%       fprintf('j = %d\n', j)
        %Composicion de los batches - del pdf
        j_start = (j-1)*GD.n_batch + 1;
        j_end = j*GD.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        % calif del batch
        [P,h,s1] = EvaluateClassifier(Xbatch, W, b);
        [LW,Lb,JW,Jb] = ComputeGradients(Xbatch, Ybatch, P, W,b, h, s1, lambda);
        W{1} = W{1} - et*JW{1}; b{1} = b{1} - et*Lb{1};
        W{2} = W{2} - et*JW{2}; b{2} = b{2} - et*Lb{2};
        %Adding the momentum
        v_W{1}=rho*v_W{1}+et*JW{1};
        v_W{2}=rho*v_W{2}+et*JW{2};
        v_b{1}=rho*v_b{1}+et*Jb{1};
        v_b{2}=rho*v_b{2}+et*Jb{2};
        W{1}=W{1}-v_W{1};
        W{2}=W{2}-v_W{2};
        b{1}=b{1}-v_b{1};
        b{2}=b{2}-v_b{2};
        
        % finish of momentum
        W={W{1},W{2}};
        b={b{1},b{2}};
    end
    [J,J1] = ComputeCost(X,Y,W,b,lambda);
    JK = [JK;J];
    W1star=W{1}; b1star=b{1}; W2star=W{2}; b2star=b{2};
     et=et*decay_rate;
end
Wstar={W1star,W2star};
bstar={b1star,b2star};
close(wb);
end

