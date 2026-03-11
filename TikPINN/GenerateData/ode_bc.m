solinit = bvpinit(linspace(0,1,10), @guess);  % 初猜
sol = bvp4c(@odefun, @bcfun, solinit);        % 调用求解器

x = linspace(0,1,500);
u = deval(sol, x);  % 插值计算解
q = 1+x.*(1-x).*sin(2*pi*x);
qu = q.*u(1,:);

plot(x, u(1,:), 'LineWidth', 2)
xlabel('x'); ylabel('y');
title('二阶ODE边值问题的数值解')
grid on

function dydx = odefun(x, u)
    dydx = [u(2);
           %-pi^2 * u(1)];
           (1+x*(1-x)*sin(2*pi*x)) * u(1)-1];
end

function res = bcfun(ua, ub)
    res = [ua(1) - 1;  % y(0) = 0
           ub(1) - 1]; % y(1) = 0
end

function g = guess(x)
    g = [x*(1-x); 1-2*x];  % 初猜
end