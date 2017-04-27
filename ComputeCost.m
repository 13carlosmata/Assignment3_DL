function [J] = ComputeCost(X,Y,W,b,lambda,L)
n=size(Y,2);
lcross=0;
tr1=(Y)';
[P,h,s] = EvaluateClassifier(X,W,b,L);
sumW={};
J2=lambda;
for i=1:n
  lcross=-log(tr1(i,:)*P(:,i))+lcross;
end
for j=1:L
    sumW{j}=sum(W{j}.^2);
end
for k=1:L
    J2=J2*sum(sumW{k});
J1=lcross/n;    
J=J1+J2;
end

