% --- Radix-2 Decimation In Frequency - Iterative approach

clear all
close all
clc

N = 32;

x = randn(1, N);
xoriginal = x;
xhat = zeros(1, N);

numStages = log2(N);

omegaa = exp(-1i * 2 * pi / N);

mulCount = 0;
sumCount = 0;
tic
M = N / 2;
for p = 1 : numStages;
    for index = 0 : (N / (2^(p - 1))) : (N - 1);
        for n = 0 : M - 1;        
            a = x(n + index + 1) + x(n + index + M + 1);
            b = (x(n + index + 1) - x(n + index + M + 1)) .* omegaa^((2^(p - 1) * n));
            x(n + 1 + index) = a;
            x(n + M + 1 + index) = b;
            mulCount = mulCount + 4;
            sumCount = sumCount + 6;
        end;
    end;
    M = M / 2;
end
xhat = bitrevorder(x);
timeCooleyTukey = toc;

tic
xhatcheck = fft(xoriginal);
timeFFTW = toc;

rms = 100 * sqrt(sum(sum(abs(xhat - xhatcheck).^2)) / sum(sum(abs(xhat).^2)));

fprintf('Time Cooley-Tukey = %f; \t Time FFTW = %f\n\n', timeCooleyTukey, timeFFTW);
fprintf('Theoretical multiplications count \t = %i; \t Actual multiplications count \t = %i\n', ...
         2 * N * log2(N), mulCount);
fprintf('Theoretical additions count \t\t = %i; \t Actual additions count \t\t = %i\n\n', ...
         3 * N * log2(N), sumCount);
fprintf('Root mean square with FFTW implementation = %.10e\n', rms);
