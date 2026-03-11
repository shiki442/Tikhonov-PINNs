clear, clc

%% q_dagger and u_dagger
solinit = bvpinit(linspace(0,1,10), @guess);  % 初猜
sol = bvp4c(@odefun, @bcfun, solinit);        % 调用求解器

x = linspace(0,1,500);
u = deval(sol, x);  % 插值计算解

q_dagger = @(x)1+x.*(1-x).*sin(2*pi*x);
u_dagger = @(x)deval(sol, x, 1)';
grad_u_x = @(x)deval(sol, x, 2)';
f_dagger = @(x)ones(size(x));

q = 1+x.*(1-x).*sin(2*pi*x);
qu = q.*u(1,:);

plot(x, u(1,:), 'LineWidth', 2)
xlabel('x'); ylabel('y');
title('二阶ODE边值问题的数值解')
grid on

%% generate data
folder = '../data/';
if ~exist(folder, 'dir')
    mkdir(folder); 
end
for delta = 0.01
    data_mat = generate_data_1d_v2(q_dagger,u_dagger,grad_u_x,f_dagger,delta);
    file_name = ['../data/example05data',num2str(100*delta,'%02d'),'.txt'];
    fprintf([file_name,' finished.\n']);
    writematrix(data_mat,file_name,'Delimiter','comma');
end


%%
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