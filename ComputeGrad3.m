function [grad_W,grad_b] = ComputeGrad3(X,Y,W,b,P,h,s,lambda,k)
n=size(Y,2);
G1=[];

for i=1:n
    %individuales
    Pi=P(:,i);
    Yt=(Y(:,i))';
    Xt=(X(:,i))';
    g=-Yt/(Yt*Pi)*(diag(Pi)-Pi*(Pi)');
    G1=[G1;g];
end
g=G1;
for j=k:-1:1
    grad_b{j}=zeros(size(W{j}));
    grad_W{j}=zeros(size(b{j}));
    gb=zeros(size(b{j}));
    gW=zeros(size(W{j}));
    gW1=zeros(size(W{j}));
    if j==1
        for j1=1:size(X,2)
            gb=gb+g(j1,:)';
            %gW=gW+g(j1,:)'*X(:,j1)';
        end
        gW1=gW1+g'*X';
    else
        for j2=1:size(h{j-1},2)
            gb=gb+g(j2,:)';
            %gW=gW+g(j2,:)'*(h{j-1}(:,j2))';
        end
        gW1=gW1+g'*h{j-1}';
    end
    grad_b{j}=gb/n;
    grad_W{j}=gW1/n+2*lambda*W{j};
    if j>1
        g=g*W{j}.*sign(h{j-1}');
    end
end

end
