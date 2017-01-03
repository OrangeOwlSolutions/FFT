clear all
close all
clc

N1 = 64;
N2 = 64;
M1 = 2048;
M2 = 2048;

x = randn(N1, N2);

%%%%%%%%%%%%%%%%%%%%%%%%%
% MATRIX MULTIPLICATION %
%%%%%%%%%%%%%%%%%%%%%%%%%
xhatcheck = Ea2Spectrum(-2 * pi * (0 : N1 - 1) / M1, -2 * pi * (0 : N2 - 1) / M2, x, 0 : (N1 - 1), 0 : (N2 - 1));

%%%%%%%%%%%%%%%%%%
% NLOGN APPROACH %
%%%%%%%%%%%%%%%%%%
xhat = recursiveCall2D(x, N1, N2, M1, M2);

100 * sqrt(sum(sum(abs(xhatcheck - xhat).^2)) / sum(sum(abs(xhatcheck).^2)))

% figure
% plot(abs(xhatcheck))
% hold on
% plot(abs(xhat), 'ro')
