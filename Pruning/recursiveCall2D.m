function xhat = recursiveCall2D(x, N1, N2, M1, M2)

xhat = zeros(N1, N2);
for k = 1 : N1,
    xhat(k, :) = recursiveCall((x(k, :)).', N1, M1);
end
for k = 1 : N2,
    xhat(:, k) = recursiveCall((xhat(:, k)).', N2, M2);
end
