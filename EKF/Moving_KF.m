function [x_ip1_g_ip1, P_ip1_g_ip1] = Moving_KF(x_i_g_i, P_i_g_i, dt, pseudoranges, sat_pos,  range,Q,R)
%% Inputs
% x_in - 12 x 1 vector of x1, y1, z1, x2, y2, z2, x1dot, y1dot, z1dot, x2dot, y2dot, z2dot positions in ECEF
% P_in - 12 x 12 matrix of state uncertainty
% dt - 1 x 1 scalar of time step between filter calls
% N is the number of visible satellites
% pseudoranges - 2 x N vector of satellite pseudoranges (corrected with
% clock bias)
%   the first row corresponds to pseudoranges to the first "car"
%   the second row corresponds to pseudoranges to the second "car"
% sat_pos - 3 x N vector of satellite positions in ECEF
% range - range measurement between "cars" on ground
N = length(pseudoranges);

%% Dynamics update
x_ip1_g_i(1:6) = x_i_g_i(1:6)+x_i_g_i(7:12)*dt; %stationary means position isn't changing
x_ip1_g_i(7:12) =x_i_g_i(7:12); %assume velocity isn't changing
F = zeros(12,12);
F(1:6,1:6) = eye(6);
F(1:6,7:12) = eye(6)*dt;
F(7:12,7:12) = eye(6);
P_ip1_g_i = F*P_i_g_i*F.'+Q; % state transition matrix is identity

%% measurement vector
z_ip1 = [range; pseudorange(1,:).'; pseudorange(2,:).'];
y = norm(x_ip1_g_i(1:3)-x_ip1_g_i(4:6));
r1_store =zeros(N,1);
r2_store = zeros(N,1);
for ind=1:N
    r1_store(ind) = norm(x_ip1_g_i(1:3)-sat_pos(:,ind));
    r2_store(ind) = norm(x_ip1_g_i(4:6)-sat_pos(:,ind));
end
y = [y; r1_store; r2_store];
prefit = z-y;

%% sensitivity matrix
H = zeros(2*N+1,12);
line_of_sight_2to1 = (x_ip1_g_i(1:3).'-x_ip1_g_i(4:6).')./norm((x_ip1_g_i(1:3).'-x_ip1_g_i(4:6).'));
H(1,:) = [-line_of_sight line_of_sight zeros(1,6)];
for ind=1:N
    r1 = -(sat_pos(:,ind)-x_ip1_g_i(1:3)).';
    r2 = -(sat_pos(:,ind)-x_ip1_g_i(4:6)).';
    H(ind+1,:) = [r1/norm(r1); r2/norm(r2); zeros(1,6)];
end

%% Kalman Gain
K = P_ip1_g_i*H.'*inv(R+H*P_ip1_g_i*H.');

%% update x and P
x_ip1_g_ip1 = x_ip1_g_i+K*prefit;
P_ip1_g_ip1 = (eye(6)-K*H)*P_ip1_g_i(eye(6)-K*H).'+K*R*K.';

    
