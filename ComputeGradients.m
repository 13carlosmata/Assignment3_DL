function [grad_W,grad_b] = ComputeGradients(X,Y,W,b,P,h,s,lambda,k)
n=size(X,2);
grad_W={};
grad_b={};
n=size(W,2);
g=-(Y-P)';
size(g)
if k>1
    for i=k:-1:2
        grad_b{i}=g;
        grad_W{i}=g'*h{i-1}'+2*lambda*W{i};
        if i>1
            size(W{i})
            g=g*W{i};
            size(g)
            size(h{i-1})
            g=g*sign(h{i});
            size(diag(s{i-1}>0))
            g=g*diag(s{i-1}>0);
        end
    end
end
grad_b{1}=g;
grad_W{1}=g'*X'+2*lambda*W{1};
end