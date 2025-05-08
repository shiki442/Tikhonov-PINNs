clear, clc

%% q_dagger and u_dagger
tmp = @(z)2.0*z-1.0;
eta = @(x,y)0.3*(3.0*tmp(x)-1.0).^2;
mu = @(x,y)exp(-9.0*tmp(x).^2-(6.0*y-2).^2);
zeta = @(x,y)0.6*tmp(x)-27.0*tmp(x).^3-243.0*tmp(y).^5;
xi = @(x,y)exp(-9.0*tmp(x).^2-9.0*tmp(y).^2);
gamma = @(x,y)exp(-(3.0*tmp(x)+1).^2-9.0*tmp(y).^2);
q_dagger = @(x,y)2.0+0.5*(eta(x,y).*mu(x,y)-zeta(x,y).*xi(x,y)-gamma(x,y));
u_dagger = @(x,y)1.0+sin(pi*x).*sin(pi*y);
grad_u_x = @(x,y)pi*cos(pi*x).*sin(pi*y);
grad_u_y = @(x,y)pi*sin(pi*x).*cos(pi*y);
laplace_u = @(x,y)-2.0*pi*pi*sin(pi*x).*sin(pi*y);

%% generate data
for delta = [0.01,0.10,0.20]
    data_mat = generate_date(q_dagger,u_dagger,grad_u_x,grad_u_y,laplace_u,delta);
    file_name = ['./data/example02data',num2str(100*delta,'%02d'),'.txt'];
    fprintf([file_name,' finished.\n']);
    writematrix(data_mat,file_name,'Delimiter','comma');
end

