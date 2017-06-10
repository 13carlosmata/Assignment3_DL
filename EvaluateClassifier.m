function [P,h,s] = EvaluateClassifier(X, W, b, L)
P={};
h={};
s={};
s{1}=W{1}*X+b{1};
h{1}=max(0,s{1});
if L > 1
    for i=2:L
        s{i}=W{i}*h{i-1}+b{i};
        h{i}=max(0,s{i});
    end
end
P=softmax(s{L});