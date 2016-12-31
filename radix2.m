clear all
close all
clc

N = 32;
M = 64;

% --- Original, non zero-padded, sequence
x = randn(1, N);
% x(5 : 32) = 0;
xoriginal = x;

% --- Zero-padding
if (N < M)
    x = [x zeros(1, M - N)];
end

x = bitrevorder(x);

numStages = log2(M);

xhat = zeros(1, M);

omegaa = exp(-1i * 2 * pi / M);
for currentStage = 1 : numStages
    butterflyOffset = 2^currentStage;
    i = 1;
    while (i <= (M - butterflyOffset / 2))
        for k = 0 : butterflyOffset / 2 - 1
            xhat(i)   = x(i) + x(i + butterflyOffset / 2) * omegaa^(k * 2^(numStages - currentStage)); 
            xhat(i+butterflyOffset / 2) = x(i) - x(i + butterflyOffset / 2) * omegaa^(k * 2^(numStages - currentStage));
            i = i + 1;
            if (k == (butterflyOffset / 2 - 1))
                i = i + butterflyOffset / 2;
            end
        end
    end
    x = xhat;
end

xhatcheck = fft(xoriginal, M);
100 * sqrt(sum(sum(abs(xhat - xhatcheck).^2)) / sum(sum(abs(xhat).^2)))

