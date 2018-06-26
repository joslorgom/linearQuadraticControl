function [Pdiff] = riccatiDynamics(t, P, Q, A, S)

N = size(A, 1);

P = reshape(P, N, N);

Pdiff = Q + A'*P + P*A - P*S*P;

Pdiff = reshape(Pdiff, N*N, 1);

end