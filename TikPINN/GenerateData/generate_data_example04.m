clear, clc

%% q_dagger and u_dagger
q_dagger = @(x)exp(x);
u_dagger = @(x)1.0+sin(pi*x);
grad_u_x = @(x)pi*cos(pi*x);
laplace_u = @(x)-pi*pi*sin(pi*x);

%% generate data
folder = '../data/';
if ~exist(folder, 'dir')
    mkdir(folder); 
end
for delta = [0.0,0.5]
    data_mat = generate_data_1d(q_dagger,u_dagger,grad_u_x,laplace_u,delta);
    file_name = ['../data/example04data',num2str(100*delta,'%02d'),'.txt'];
    fprintf([file_name,' finished.\n']);
    writematrix(data_mat,file_name,'Delimiter','comma');
end


