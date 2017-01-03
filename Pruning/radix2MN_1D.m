function xhat = recursiveCall(x, N, M)

x = [x(1 : 2 : N) x(2 : 2 : N)];

omegaa = exp(-1i * 2 * pi / M);

n = 0 : N / 2 - 1;
m = 0 : N / 2 - 1;
[NN, MM] = meshgrid(n, m);

WM2 = exp(-1i * 2 * pi * NN .* MM / (M / 2));

if (N > 2)
%     xhatfirst   = WM2 * x(1 : N / 2) + diag(omegaa.^(n)) * WM2 * x(N / 2 + 1 : N);
%     xhatsecond  = WM2 * (omegaa.^(N * n) .* x(1 : N / 2).').' + diag(omegaa.^(n + N / 2)) * WM2 * (omegaa.^(N * n) .* x(N / 2 + 1 : N).').';
    xhatfirst   = recursiveCall(x(1 : N / 2), N / 2, M / 2) + diag(omegaa.^(n)) * recursiveCall(x(N / 2 + 1 : N), N / 2, M / 2);
    xhatsecond  = recursiveCall((omegaa.^(N * n) .* x(1 : N / 2)).', N / 2, M / 2) + diag(omegaa.^(n + N / 2)) * recursiveCall((omegaa.^(N * n) .* x(N / 2 + 1 : N)).', N / 2, M / 2);
    xhat = [xhatfirst.' xhatsecond.'].';
else
    n = 0 : N - 1;
    m = 0 : N - 1;
    [NN, MM] = meshgrid(n, m);
    xhat = exp(-1i * 2 * pi * NN .* MM / M) * x.';
end
