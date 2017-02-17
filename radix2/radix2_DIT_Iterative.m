% --- Radix-2 Decimation In Time - Iterative approach

clear all
close all
clc

N = 32;

x = randn(1, N);
xoriginal = x;
x = bitrevorder(x);
xhat = zeros(1, N);

numStages = log2(N);

omegaa = exp(-1i * 2 * pi / N);

mulCount = 0;
sumCount = 0;
tic
for p = 1 : numStages
    alpha = 2^(p - 1);
    butterflyStart = 1;
    while (butterflyStart <= (N - alpha))
        for interButterflyIndex = 0 : alpha - 1
            xhat(butterflyStart)          = x(butterflyStart) + x(butterflyStart + alpha) * omegaa^(interButterflyIndex * 2^(numStages - p)); 
            xhat(butterflyStart + alpha)  = x(butterflyStart) - x(butterflyStart + alpha) * omegaa^(interButterflyIndex * 2^(numStages - p));
            mulCount = mulCount + 4;
            sumCount = sumCount + 6;
            butterflyStart = butterflyStart + 1;
            if (interButterflyIndex == (alpha - 1))
                butterflyStart=butterflyStart + alpha;
            end;
        end;
    end;
    x = xhat;
end;
timeCooleyTukey = toc;

tic
xhatcheck = fft(xoriginal, N);
timeFFTW = toc;

rms = 100 * sqrt(sum(sum(abs(xhat - xhatcheck).^2)) / sum(sum(abs(xhat).^2)));

fprintf('Time Cooley-Tukey = %f; \t Time FFTW = %f\n\n', timeCooleyTukey, timeFFTW);
fprintf('Theoretical multiplications count \t = %i; \t Actual multiplications count \t = %i\n', ...
         2 * N * log2(N), mulCount);
fprintf('Theoretical additions count \t\t = %i; \t Actual additions count \t\t = %i\n\n', ...
         3 * N * log2(N), mulCount);
fprintf('Root mean square with FFTW implementation = %.10e\n', rms);
