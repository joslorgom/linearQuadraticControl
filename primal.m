function [xT, x] = primal(u, x0, A, B, tout)

options = odeset('RelTol', 1e-8);

[~, x] = ode45(@(t, x) A*x + B*interp1(tout, u, t)', tout, x0, options);

xT = x(end, :)';
    
end
