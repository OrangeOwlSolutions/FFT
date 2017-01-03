clear all
close all
clc

N = 128;
M = 4096;

x = randn(1, N);

xoriginal = x;

%%%%%%%%%%%%%%%%%%%%%%%%%
% MATRIX MULTIPLICATION %
%%%%%%%%%%%%%%%%%%%%%%%%%
n = 0 : N - 1;
m = 0 : N - 1;
[NN, MM] = meshgrid(n, m);
xhatcheck = exp(-1i * 2 * pi * NN .* MM / M) * xoriginal.';

%%%%%%%%%%%%%%%%%%
% NLOGN APPROACH %
%%%%%%%%%%%%%%%%%%
xhat = recursiveCall(xoriginal.', N, M);

100 * sqrt(sum(sum(abs(xhatcheck(1 : N) - xhat(1 : N)).^2)) / sum(sum(abs(xhatcheck(1 : N)).^2)))

figure
plot(abs(xhatcheck))
hold on
plot(abs(xhat), 'ro')
