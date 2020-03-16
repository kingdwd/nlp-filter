function [x_ip1_g_ip1, P_ip1_g_ip1] = Stationary_KF(x_i_g_i, P_i_g_i, pseudoranges1, pseudoranges2, sat_pos_t1, sat_pos_t2, range,Q,R)
%% Inputs
% x_in - 6 x 1 vector of x1, y1, z1, x2, y2, z2 positions in ECEF
% P_in - 6 x 6 matrix of state uncertainty
% dt - 1 x 1 scalar of time step between filter calls
% N is the number of visible satellites
% pseudoranges - 2 x N vector of satellite pseudoranges
%   the first row corresponds to pseudoranges to the first "car"
%   the second row corresponds to pseudoranges to the second "car"
% sat_pos - 3 x N vector of satellite positions in ECEF
% range - range measurement between "cars" on ground
N1 = length(pseudoranges1);
N2 = length(pseudoranges2);

%% Dynamics update
x_ip1_g_i = x_i_g_i; %stationary means position isn't changing
P_ip1_g_i = P_i_g_i+Q; % state transition matrix is identity

%% measurement vector
z_ip1 = [range; pseudoranges1.'; pseudoranges2.'];
y = norm(x_ip1_g_i(1:3)-x_ip1_g_i(4:6));
r1_store =zeros(N1,1);
r2_store = zeros(N2,1);
for ind=1:N1
    r1_store(ind) = norm(x_ip1_g_i(1:3)-sat_pos_t1(:,ind));
end
for ind=1:N2
     r2_store(ind) = norm(x_ip1_g_i(4:6)-sat_pos_t2(:,ind));
end
y = [y; r1_store; r2_store];
prefit = z_ip1-y;

%% sensitivity matrix
H = zeros(N1+N2+1,6);
line_of_sight = (x_ip1_g_i(1:3).'-x_ip1_g_i(4:6).')./norm((x_ip1_g_i(1:3).'-x_ip1_g_i(4:6).'));
H(1,:) = [line_of_sight -line_of_sight];
for ind=1:N1
    r1 = -(sat_pos_t1(:,ind)-x_ip1_g_i(1:3)).';
    H(ind+1,:) = [r1/norm(r1) zeros(1,3)];
end
for ind=1:N2
    r2 = -(sat_pos_t2(:,ind)-x_ip1_g_i(4:6)).';
    H(ind+N1,:) = [zeros(1,3) r2/norm(r2)];
end

%% Kalman Gain
K = P_ip1_g_i*H.'*inv(R+H*P_ip1_g_i*H.');

%% update x and P
x_ip1_g_ip1 = x_ip1_g_i+K*prefit;
P_ip1_g_ip1 = (eye(6)-K*H)*P_ip1_g_i*(eye(6)-K*H).'+K*R*K.';

    
