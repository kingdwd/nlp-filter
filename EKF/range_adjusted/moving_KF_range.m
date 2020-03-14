function [x_ip1_g_ip1, P_ip1_g_ip1] = moving_KF_range(x_i_g_i, P_i_g_i, pseudoranges1, pseudoranges2, sat_pos_t1, sat_pos_t2, range,Q,R,dt, v1,v2)
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
x_ip1_g_i = x_i_g_i+[v1; 0; 0; v2; 0; 0]; %stationary means position isn't changing
x_ip1_g_i(4) = x_ip1_g_i(4)+dt*x_ip1_g_i(5); %update clock drift
x_ip1_g_i(9) = x_ip1_g_i(9)+dt*x_ip1_g_i(10); %update clock drift
F = eye(10);
F(4,5) = dt; %accounting for drift rate in clock bias
F(9,10) = dt; %accounting for drift rate in clock bias
P_ip1_g_i = F*P_i_g_i*F.'+Q; % update state transition matrix

%% measurement vector
z_ip1 = [range; pseudoranges1.'; pseudoranges2.']; %measurement vector - range, pseudoranges car 1, pseudoranges car 2

%calculate estimated range
[lat, long, h] = ECEF_to_LLA(x_ip1_g_i(1),x_ip1_g_i(2),x_ip1_g_i(3)); %get lat long
R_rot = [-sind(long) cosd(long) 0; -sind(lat)*cosd(long) -sind(lat)*sind(long) cosd(lat); cosd(lat)*cosd(long) cosd(lat)*sind(long) sind(lat)]; %get rotation from ENU (compass measure) to ECEF
%ecef to enu
temp = R_rot*(x_ip1_g_i(1:3)-x_ip1_g_i(6:8));
range_est = norm(temp(1:2));

%calculate pseudorange vector
r1_store =zeros(N1,1);
r2_store = zeros(N2,1);
for ind=1:N1
    r1_store(ind) = norm(x_ip1_g_i(1:3)-sat_pos_t1(:,ind))+x_ip1_g_i(4);
end
for ind=1:N2
     r2_store(ind) = norm(x_ip1_g_i(6:8)-sat_pos_t2(:,ind))+x_ip1_g_i(9);
end

%form model measurement and prefit
y = [range_est; r1_store; r2_store];
prefit = z_ip1-y;

%% sensitivity matrix
%initialize
H = zeros(N1+N2+1,10);

%range sensitivity
diff_x = x_ip1_g_i(1:3)-x_ip1_g_i(6:8);
Rot_red = R_rot(1:2,:);

H(1,1) = (dot(R_rot(1,:),diff_x.')*R_rot(1,1)+dot(R_rot(2,:),diff_x.')*R_rot(2,1))/norm(Rot_red*diff_x);
H(1,6) = -H(1,1);
H(1,2) = (dot(R_rot(1,:),diff_x.')*R_rot(1,2)+dot(R_rot(2,:),diff_x.')*R_rot(2,2))/norm(Rot_red*diff_x);
H(1,7) = -H(1,2);
H(1,3) = (dot(R_rot(1,:),diff_x.')*R_rot(1,3)+dot(R_rot(2,:),diff_x.')*R_rot(2,3))/norm(Rot_red*diff_x);
H(1,8) = -H(1,3);

%pseudorange sensitivity
%car 1
for ind=1:N1
    r1 = -(sat_pos_t1(:,ind)-x_ip1_g_i(1:3)).';
    H(ind+1,:) = [r1/norm(r1) 1 0 zeros(1,5)];
end

%car 2
for ind=1:N2
    r2 = -(sat_pos_t2(:,ind)-x_ip1_g_i(6:8)).';
    H(ind+N1,:) = [zeros(1,5) r2/norm(r2) 1 0];
end

%% Kalman Gain
K = P_ip1_g_i*H.'/(R+H*P_ip1_g_i*H.');

%% update x and P
x_ip1_g_ip1 = x_ip1_g_i+K*prefit;
P_ip1_g_ip1 = (eye(10)-K*H)*P_ip1_g_i*(eye(10)-K*H).'+K*R*K.';

    
