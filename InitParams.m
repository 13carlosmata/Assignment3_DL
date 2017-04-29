function [W,b] = InitParams(d,m,K,L)
mean=0;
std=0.001;
W={};
b={};
%Init of cells
for i=1:L-1
    if i==1
        W{i} = mean + std.*randn(m(i),d);
        b{i} = mean + std.*randn(m(i),1);
    else
        W{i} = mean + std.*randn(m(i),m(i-1));
        b{i} = mean + std.*randn(m(i),1);
    end
end
if L~=1
    W{L} = mean + std.*randn(K,m(i));
    b{L} = mean + std.*randn(K,1);
else
    W{1} = mean + std.*randn(K,d);
    b{1} = mean + std.*randn(K,1);
end
end