clear all
close all
clc

N = 1024;
M = 1024;

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

operationCounter = 0;
omegaa = exp(-1i * 2 * pi / M);
for currentStage = 1 : numStages
    butterflyOffset = 2^(currentStage - 1);
    i = 1;
    while (i <= (M - butterflyOffset))
        for k = 0 : butterflyOffset - 1
            xhat(i)   = x(i) + x(i + butterflyOffset) * omegaa^(k * 2^(numStages - currentStage)); 
            xhat(i + butterflyOffset) = x(i) - x(i + butterflyOffset) * omegaa^(k * 2^(numStages - currentStage));
            i = i + 1;
            operationCounter = operationCounter + 2;
            if (k == (butterflyOffset - 1))
                i = i + butterflyOffset;
            end
        end
    end
    x = xhat;
end

xhatcheck = fft(xoriginal, M);
100 * sqrt(sum(sum(abs(xhat - xhatcheck).^2)) / sum(sum(abs(xhat).^2)))

fprintf('Number of operations = %f; M * log2(M) = %f\n', operationCounter, M * log2(M));
