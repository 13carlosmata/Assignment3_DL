function [Wstar,bstar,JK,u,V] = BatchNorm(X, Y, GD, W,b, lambda, L)
N=size(X,2);
[J] = ComputeCost(X,Y,W,b,lambda,L);
JK=[J];
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
        % mean and variances for un-normalized scores
        re=s{1};
        for kun=1:L
            for l=1:size(s{kun},2)
                u{kun}=sum(s{kun}(:,l));
            end
            u{kun}=u{kun}/size(s{kun},2);
            ml=size(s{kun},1);
%             Vp=[];
%             for jun=1:ml
%                 for ix=1:size(s{kun},2)
%                     v=sum((s{kun}(jun,ix)-u{kun}).^2);
%                 end
%                 v=v/size(s{kun},2);
%                 Vp=[Vp,v];
%             end
%             V{kun}=Vp;
            V{kun}=(var(s{kun},0,2))*(size(s{kun},2)-1) / size(s{kun},2);
            s{kun}=(diag(V{kun}+8.9e-12)^-0.5)*(s{kun}-u{kun});
        end
        %%%%%%%%%%%%%%%%%%%%%       
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
    Wstar=W;
    bstar=b;
    et=et*decay_rate;
end
close(wb);
end

