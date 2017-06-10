function [Wstar,bstar,JK,JK_val] = MiniBatchGD(X, Y,valX,valY, GD, W,b, lambda, L)
N=size(X,2);
[J] = ComputeCost(X,Y,W,b,lambda,L);
[J2] = ComputeCost(valX,valY,W,b,lambda,L);
JK=[J];
JK_val=[J2];
decay_rate=0.95;
rho=0.9;
et=GD.eta;
for init=1:L
    v_W{init}=zeros(size(W{init})); 
    v_b{init}=zeros(size(b{init}));    
end
wb = waitbar(0,'1','Name','Minibatch progress');
for i=1:GD.n_epochs
    waitbar(i/GD.n_epochs,wb,strcat('Epoch: ',num2str(i)));
    for j=1:N/GD.n_batch
        j_start = (j-1)*GD.n_batch + 1;
        j_end = j*GD.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        [P,h,s] = EvaluateClassifier(Xbatch, W, b, L);
        [grad_W,grad_b] = ComputeGrad3(Xbatch,Ybatch,W,b,P,h,s,lambda,L);
        for k=1:L
            W{k} = W{k}-et*grad_W{k};
            b{k} = b{k}-et*grad_b{k};
            %starts momentum
            v_W{k} = rho*v_W{k} + et*grad_W{k};
            v_b{k} = rho*v_b{k} + et*grad_b{k};
            W{k} = W{k} - v_W{k};
            b{k} = b{k} - v_b{k};
            %finish momentum            
        end
    end
    [J] = ComputeCost(X,Y,W,b,lambda,L);
    JK=[JK;J];
    [J2] = ComputeCost(valX,valY,W,b,lambda,L);
    JK_val=[JK_val;J2];
    Wstar=W;
    bstar=b;
    et=et*decay_rate;
end
close(wb);
end

