function data_mat = generate_date_1d(q_dagger,u_dagger,grad_u_x,...
    laplace_u,noise_level)
rng(2468);
n_samples = 50000;

%% interior
n_samples_int = n_samples;
xq = rand(n_samples_int,1);
int_point = xq;
% u_int: solution
u_int_val = u_dagger(xq);
% f: source density
laplace_u_int_val = laplace_u(xq);
q_val = q_dagger(xq);
f_val = -laplace_u_int_val+q_val.*u_int_val;

%% boundary
n_samples_boundary = n_samples / 2;

bdy_point = [zeros(n_samples_boundary,1);ones(n_samples_boundary,1)];
normal_vec = [-ones(n_samples_boundary,1);ones(n_samples_boundary,1)];
% u_bdy: solution
u_bdy_val = u_dagger(bdy_point);
% g_val: boundary flux
u_grad = [grad_u_x(bdy_point)];
g_val = sum(u_grad.*normal_vec,2);

%% output data
% add noise
scale = norm(u_int_val,"inf");
m_int = u_int_val+noise_level*scale*randn(n_samples,1);
m_bdy = u_bdy_val+noise_level*scale*randn(n_samples,1);
% data matrix
data_mat = [int_point,bdy_point,normal_vec,m_int,m_bdy,f_val,g_val,u_int_val,q_val];
rand_index = randperm(n_samples);
data_mat = data_mat(rand_index,:);
end

