function [p] = dual(pT, u, A, B, tout)

options = odeset('RelTol', 1e-8);

u = flipud(u);

[~, p] = ode45(@(t, p) A'*p + interp1(tout, u, t)', tout, pT, options);

p = flipud(p*B);

end
