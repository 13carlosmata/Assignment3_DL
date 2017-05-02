% function acc = ComputeAccuracy(X,Y,W,b,L)
% suma = 0;
% [P,h,s] = EvaluateClassifier(X, W, b, L);
% for i=1:size(X,2)
%     [value,index] = max(P(:,i));
%     if Y(index,i) == 1
%         suma = suma + 1;
%     end
% end
% acc=suma/size(X,2)*100;
% 
% 
% 
function acc = ComputeAccuracy(X, y, W, b,L)

% Probalistic output times ground truth gives estemiated answer
% and compare the values with the most possible answer
[p,h,s1] = EvaluateClassifier(X, W, b,L);
diff = max(p) - sum(p.*y,1);
acc=nnz(diff==0)/size(y,2)*100;

end