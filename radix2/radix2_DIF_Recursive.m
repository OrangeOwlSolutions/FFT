% --- Radix-2 Decimation In Frequency - Iterative approach

function y = radix2_DIF_Recursive(x)

global sumCount mulCount

N = length(x);
phasor = exp(-2 * pi * 1i / N) .^ (0 : N / 2 - 1);

if N == 1
    y = x;
else
    y_top       = radix2_DIF_Recursive(x(1: 2 : (N - 1)));
    y_bottom    = radix2_DIF_Recursive(x(2 : 2 : N));
    z           = phasor .* y_bottom;
    y           = [y_top + z, y_top - z];
    sumCount    = sumCount + 6 * (N / 2);
    mulCount    = mulCount + 4 * (N / 2);
end