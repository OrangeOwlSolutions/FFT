% --- Radix-2 Decimation In Frequency - Iterative approach

clear all
close all
clc

% --- N should be a power of 4
N = 1024;

% x = randn(1, N);
x = zeros(1, N);
x(1 : 10) = 1;
xoriginal = x;
xhat = zeros(1, N);

numStages = log2(N) / 2;

W = exp(-1i * 2 * pi * (0 : N - 1) / N);
omegaa = exp(-1i * 2 * pi / N);

mulCount = 0;
sumCount = 0;

M = N / 4;
for p = 1 : numStages;
    for index = 0 : (N / (4^(p - 1))) : (N - 1);
        for n = 0 : M - 1;   
            a =  x(n + index + 1) +      x(n + index + M + 1) + x(n + index + 2 * M + 1) +      x(n + index + 3 * M + 1);
            b = (x(n + index + 1) -      x(n + index + M + 1) + x(n + index + 2 * M + 1) -      x(n + index + 3 * M + 1)) .* omegaa^(2 * (4^(p - 1) * n));
            c = (x(n + index + 1) - 1i * x(n + index + M + 1) - x(n + index + 2 * M + 1) + 1i * x(n + index + 3 * M + 1)) .* omegaa^(1 * (4^(p - 1) * n));
            d = (x(n + index + 1) + 1i * x(n + index + M + 1) - x(n + index + 2 * M + 1) - 1i * x(n + index + 3 * M + 1)) .* omegaa^(3 * (4^(p - 1) * n));
            x(n + 1 + index) = a;
            x(n + M + 1 + index) = b;
            x(n + 2 * M + 1 + index) = c;
            x(n + 3 * M + 1 + index) = d;
            mulCount = mulCount + 3;
            sumCount = sumCount + 8;
        end;
    end;
    M = M / 4;
end

xhat = bitrevorder(x);
   
tic
xhatcheck = fft(xoriginal);
timeFFTW = toc;

rms = 100 * sqrt(sum(sum(abs(xhat - xhatcheck).^2)) / sum(sum(abs(xhat).^2)));

fprintf('Theoretical multiplications count \t = %i; \t Actual multiplications count \t = %i\n', ...
         (3 / 8) * N * log2(N), mulCount);
fprintf('Theoretical additions count \t\t = %i; \t Actual additions count \t\t = %i\n\n', ...
         N * log2(N), sumCount);
fprintf('Root mean square with FFTW implementation = %.10e\n', rms);
