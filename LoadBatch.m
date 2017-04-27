function [X, Y, y] = LoadBatch(filename)
load('batches.meta.mat')
A = load (filename);
I = reshape(A.data',32,32,3,10000);
I = permute(I, [2,1,3,4]);
%montage(I(:,:,:,:),'Size',[5,5]);
Y = zeros(10,10000);
X = im2double(I);
for i=1:10000
   Y((A.labels(i)+1),i)=1;
end
y={};
for i=1:10000
   y{i}= label_names(A.labels(i)+1);
end
y=y';
X=reshape(X,3072,10000);    %trainX with size dxn  -> 3072x10000
end

