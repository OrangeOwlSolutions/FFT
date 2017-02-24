% --- Radix-2 Decimation In Frequency - Iterative approach

clear all
close all
clc

global sumCount mulCount

N = 32;

x = randn(1, N);

sumCount = 0;
mulCount = 0;

tic
xhat = radix2_DIF_Recursive(x);
timeCooleyTukey = toc;

tic
xhatcheck = fft(x);
timeFFTW = toc;

rms = 100 * sqrt(sum(sum(abs(xhat - xhatcheck).^2)) / sum(sum(abs(xhat).^2)));

fprintf('Time Cooley-Tukey = %f; \t Time FFTW = %f\n\n', timeCooleyTukey, timeFFTW);
fprintf('Theoretical multiplications count \t = %i; \t Actual multiplications count \t = %i\n', ...
         2 * N * log2(N), mulCount);
fprintf('Theoretical additions count \t\t = %i; \t Actual additions count \t\t = %i\n\n', ...
         3 * N * log2(N), sumCount);
fprintf('Root mean square with FFTW implementation = %.10e\n', rms);
