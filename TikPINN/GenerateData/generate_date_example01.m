clear, clc

%% q_dagger and u_dagger
q_dagger = @(x,y)1.0+0.5*sin(pi*x).*sin(pi*y);
u_dagger = @(x,y)1.0+sin(pi*x).*sin(pi*y);
grad_u_x = @(x,y)pi*cos(pi*x).*sin(pi*y);
grad_u_y = @(x,y)pi*sin(pi*x).*cos(pi*y);
laplace_u = @(x,y)-2.0*pi*pi*sin(pi*x).*sin(pi*y);

%% generate data
folder = '../data/';
if ~exist(folder, 'dir')
    mkdir(folder); 
end
for delta = [0.01,0.10,0.20,0.50]
    data_mat = generate_date(q_dagger,u_dagger,grad_u_x,grad_u_y,laplace_u,delta);
    file_name = ['../data/example01data',num2str(100*delta,'%02d'),'.txt'];
    fprintf([file_name,' finished.\n']);
    writematrix(data_mat,file_name,'Delimiter','comma');
end

