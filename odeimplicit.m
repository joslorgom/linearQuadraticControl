function [x] = odeimplicit(f, tspan, x0)

tol = 1e-10;
theta = 0.5;

Nt = length(tspan);
x = zeros(Nt, length(x0));
x(1, :) = x0';

for i = 2:length(tspan)
    k = 0;
    xcurr = x(i-1)';
    xnext = xcurr;
    error = 10*tol;
    while error > tol
        xth = (1-theta)*xcurr + theta*xnext;
        tth = (1-theta)*tspan(i-1) + theta*tspan(i);
        xnexta = xnext;
        xnext = xcurr + (tspan(i) - tspan(i-1)) * f(tth, xth);
        error = (xnext - xnexta)' * (xnext - xnexta);
        k = k + 1;
    end
    x(i) = xnext;
end

end