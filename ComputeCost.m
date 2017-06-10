function [J] = ComputeCost(X,Y,W,b,lambda,L)
n=size(Y,2);
lcross=0;
[P,h,s] = EvaluateClassifier(X,W,b,L);
sumW={};
J2=lambda;
for i=1:n
  lcross=-log(Y(:,i)'*P(:,i))+lcross;
end
for j=1:L
    sumW{j}=sum(W{j}.^2);
end
for k=1:L
    J2=J2*sum(sumW{k});
end
J1=lcross/n;    
J=J1+J2;
end
