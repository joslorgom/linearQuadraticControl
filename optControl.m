clc;
close all;
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the optimal control for the cost function
% J = beta1/2*integral(u'(t)*u(t)) + beta2/2*|| xav(T) - Xd ||^2 +
%       + beta3/2*integral(||xav - xd||^2)
% subject to:
% dx/dt = A*x + B*u
% x(0) = x0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Size of state vector
N = 2;
% Size of control vector 
M = 1;

% Initial condition
x0 = [1; 1];

% Target state
xdT = ones(N, 1); 
xd = [1, 1]';

% Parameter beta for the cost function
beta1 = 1;
beta2 = 0;
beta3 = 1;

% Time span
tspan = [0, 20];

% Number of time steps
Nt = 1000;

% Maximum number of iterations
Nmax = 2000;

% Tolerance
tol = 1e-8;

%A = -[2 -1; -1 2];
A = [0 1; -1, -2];
%B = [1 0; 0 1];
B = [0; 1];
%B = [1; 0];

A = [0 1; 0 0];
B = [0; 1];

x0 = [0.1; 0];

interpMethod = 'pchip';
gradientMethod = 'cg';

epsilon = 0.0001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T0 = tspan(1);
T = tspan(2);

% Time vector
tout = linspace(T0, T, Nt)';
% t1 = linspace(T0, 0.1*T, 10000)';
% t2 = linspace(0.1*T, 0.9*T, 100)';
% t3 = linspace(0.9*T, T, 10000)';
% tout = [t1; t2(2:end-1); t3];
% Nt = length(tout);

xd = [cos(2*pi*tout'); sin(2*pi*tout')];

% Initial control
u = zeros(Nt, M);

% Iteration counter
iter = 0;

% Compute b in gradient = A*u-b for CG method
% Free dynamics solution
[xhT, xh] = primal(zeros(Nt, M), x0, A, B, tout);
b = dual(beta2*(xdT - xhT), beta3*(xd' - xh), A, B, tout);

if strcmpi(gradientMethod, 'cg')
    % Compute A*u
    % Zero initial condition solution
    [xuT, xu] = primal(u, zeros(N, 1), A, B, tout);
    Au = dual(beta2*xuT, beta3*xu, A, B, tout) + beta1*u;

    xua = xu;

    % Compute initial gradient and store previous gradient
    % ga = Au - b; 
    % g = ga;

    g = Au - b;

    % Initial gradient norm
    % g0L2 = integral(@(t) interp1(tout, sum(g.*g, 2), t), T0, T);
    % gaL2 = g0L2;
    % gL2 = g0L2;

    gL2 = integral(@(t) interp1(tout, sum(g.*g, 2), t), T0, T);

    % Residual b-A*u
    r = -g;
    % Residual norm
    rn = sqrt( integral(@(t) interp1(tout, sum(r.*r, 2), t, interpMethod), T0, T) );
    
elseif strcmpi(gradientMethod, 'sd')
    rn = 10*tol;
end

J = 0;
dJ = 10*tol;

% && dJ > tol
while (rn > tol && iter <= Nmax)
    
    if strcmpi(gradientMethod, 'cg')
    
        % A*u
        [xuT, xu] = primal(r, zeros(N, 1), A, B, tout);
        w = dual(beta2*xuT, beta3*xu, A, B, tout) + beta1*r;

        % alpha
    %     alpha = integral(@(t) interp1(tout, sum(g.*g, 2), t, interpMethod), T0, T)/...
    %         integral(@(t) interp1(tout, sum(r.*w, 2), t, interpMethod), T0, T);
        alpha = gL2/integral(@(t) interp1(tout, sum(r.*w, 2), t, interpMethod), T0, T);

        % Update control
        u = u + alpha*r;

        % Store previous gradient
        %ga = g;
        gaL2 = gL2;
        % Update gradient
        g = g + alpha*w;
        % Compute gradient norm
        gL2 = integral(@(t) interp1(tout, sum(g.*g, 2), t, interpMethod), T0, T);

        % gamma
        gamma = gL2/gaL2;

        % Update residual
        r = -g + gamma*r;

        % Compute residual norm
        rn = sqrt( integral(@(t) interp1(tout, sum(r.*r, 2), t, interpMethod), T0, T) );

        xua = xua + alpha*xu;
        x = xh + xua;
        xuT = x(end, :)';
        
    elseif strcmpi(gradientMethod, 'sd')
        
        % A*u
        [xuT, xu] = primal(u, zeros(N, 1), A, B, tout);
        Au = dual(beta2*xuT, beta3*xu, A, B, tout) + beta1*u;
        
        g = Au - b;
        
        u = u - epsilon*g;
        
        rn = sqrt( integral(@(t) interp1(tout, sum(g.*g, 2), t, interpMethod), T0, T) );
        
        x = xh + xu;
        
    end
        
    % Update iteration counter
    iter = iter + 1;
    
    Ja = J;
    % Compute functional
    Ju = 0.5 * beta1 * integral(@(t) interp1(tout, sum(u.*u, 2), t, interpMethod), T0, T);
    JT = 0.5 * beta2 * (xuT + xhT - xdT)' * (xuT + xhT - xdT);
    Jx = 0.5 * beta3 * integral(@(t) interp1(tout, sum((x -xd').*(x - xd'), 2), t, interpMethod), T0, T);
    J = JT + Ju + Jx;
    
    dJ = abs( J - Ja);

    % Print status message
    fprintf("Iteration %i - Error %g - Cost %g\n", iter, rn, J);
    
end

[~, x2] = primal(u, x0, A, B, tout);

C = A\B;
utp = -(beta3*(C'*C) + beta1*eye(M))\(xd'*C)*beta3;
utp = utp';
xtp = -C*utp;

figure(1)
subplot(1, 3, 1)
plot(tout, u(:, 1), 'r')
hold on
plot(tout, utp(1)*ones(size(tout)), 'b')
subplot(1, 3, 2)
plot(tout, x(:, 1), 'r')
hold on
plot(tout, xtp(1)*ones(size(tout)), 'b')
subplot(1, 3, 3)
plot(tout, x(:, 2), 'r')
hold on
plot(tout, xtp(2)*ones(size(tout)), 'b')