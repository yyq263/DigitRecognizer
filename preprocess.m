clear 
clc
%% Load data
data_train = csvread('train_nolabels.csv');
data_test = csvread('test_nolabels.csv');
%% Verify the numbers represented by pixels
pick = 1;
v = data_test(pick, :);
V = reshape(v(1:end), [28,  28])';
imshow(V);
%fprintf('digit=%d\n', v(1));
%% Reshape the data
X = data_train(:, 2:end);
[nSample, nFeature] = size(X);
[TSample, TFeature] = size(data_test);
X_nor = double(X > 0);
data_test_nor = double(data_test > 0);
X_re = reshape(X_nor', [28, nSample * nFeature / 28])';
T_re = reshape(data_test_nor', [28, TSample * TFeature / 28])';
y = data_train(:, 1);
%imshow(X_re(29:56, :))
%% Do pca for training data
nLine = size(X_re, 1);
D_set = zeros(28, nSample);
X_de = zeros(size(X_re));
j = 1;
for i = 1: 28 : nLine
    ex = X_re(i:i+27, :);
    A = ex * ex';
    [E,D] = eig(A);
    D_set(:, j) = diag(D);
    X_de(i:i+27, :) = E' * ex; % decorrelate
    j = j + 1;
end
%% Do pca for test data
TLine = size(T_re, 1);
T_set = zeros(28, TSample);
T_de = zeros(size(T_re));
j = 1;
for i = 1 : 28 : TLine
    tx = T_re(i:i+27, :);
    A = tx * tx';
    [E, D] = eig(A);
    T_set(:, j) = diag(D);
    T_de(i:i+27, :) = E' * tx; % decorrelate
    j = j + 1;
end
%% 1-NN, predict T_set based on D_set
%TT = T_de(((pick-1) * 28 + 1) : ((pick-1) * 28 + 28), :);
y_test = zeros(TSample, 1);
errs = zeros(nSample,1);
for j = 1 : TSample
    TT = T_de(((j-1) * 28 + 1) : ((j-1) * 28 + 28), :);
    for i = 1 : nSample
        DD = X_de(((i-1) * 28 + 1) : ((i-1) * 28 + 28), :);
        errs(i) = sqrt((TT(:) - DD(:))' * (TT(:) - DD(:)));
    end
    [min_error, I] = min(errs);
    y_test(j) = y(I);
end

[errs_s, I] = sort(errs);
fprintf('%d\n', y(I(1))); % Finally 1-NN.
%fprintf('%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n', y(I(1)), y(I(2)), y(I(3)), y(I(4)), y(I(5)), y(I(6)), y(I(7)), y(I(8)), y(I(9)), y(I(10)), y(I(11)), y(I(12)), y(I(13)), y(I(14)), y(I(15)));
%% Write data to files
fd = fopen('y_test', 'w');
for i = 1 : TSample
   fprintf(fd, '%d\n', y(i)); 
end
fclose(fd);
