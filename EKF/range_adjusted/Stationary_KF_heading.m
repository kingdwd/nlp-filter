function [x_ip1_g_ip1, P_ip1_g_ip1] = Stationary_KF_heading(x_i_g_i, P_i_g_i, pseudoranges1, pseudoranges2, sat_pos_t1, sat_pos_t2, range,Q,R,dt, theta)
%% Inputs
% x_i_g_i - 10 x 1 vector of x1, y1, z1, bias 1, bias rate 1, x2, y2, z2, bias 2, bais rate 2 positions in ECEF
% P_i_g_i - 10 x 10 matrix of state uncertainty
% pseudoranges1 - 2 x N1 vector of satellite pseudoranges for first "car"
% pseudoranges2 - 2 x N2 vector of satellite pseudoranges for second "car"
% sat_pos - 3 x N vector of satellite positions in ECEF
% range - range measurement between "cars" on ground
% Q - process noise matirx (10 x 10)
% R - measurement noise matrix (2+N1+N2) x (2+N1+N2)
% dt - time step between measurements
% heading - compas angle measured between "cars" on ground - measured from
% car 2 (samsung) to car 1 (onyx)

%satellites in view for each car
N1 = length(pseudoranges1);
N2 = length(pseudoranges2);

%% Dynamics update
x_ip1_g_i = x_i_g_i; %stationary means position isn't changing
x_ip1_g_i(4) = x_ip1_g_i(4)+dt*x_ip1_g_i(5); %update clock drift
x_ip1_g_i(9) = x_ip1_g_i(9)+dt*x_ip1_g_i(10); %update clock drift
F = eye(10); %identity
F(4,5) = dt; %accounting for drift rate in clock bias
F(9,10) = dt; %accounting for drift rate in clock bias
P_ip1_g_i = F*P_i_g_i*F.'+Q; % update state transition matrix

%% measurement vector
z_ip1 = [range; theta; pseudoranges1.'; pseudoranges2.']; %measurement vector - range, heading, pseudoranges car 1, pseudoranges car 2

%calculate range
[lat, long, h] = ECEF_to_LLA(x_ip1_g_i(1),x_ip1_g_i(2),x_ip1_g_i(3)); %get lat long
R_rot_ecef_enu = [-sind(long) cosd(long) 0; -sind(lat)*cosd(long) -sind(lat)*sind(long) cosd(lat); cosd(lat)*cosd(long) cosd(lat)*sind(long) sind(lat)]; %get rotation from ENU (compass measure) to ECEF
%ecef to enu
temp = R_rot_ecef_enu*(x_ip1_g_i(1:3)-x_ip1_g_i(6:8));
range_est = norm(temp(1:2));

%calculate theta
R_rot = inv(R_rot_ecef_enu); %want enu to ecef
x_temp_ENU = 1/range_est*inv(R_rot)*(x_ip1_g_i(1:3)-x_ip1_g_i(6:8)); %onyx - samsung because samsung to onyx heading measured
theta_est = rad2deg(atan2(x_temp_ENU(2),x_temp_ENU(1)));

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
y = [range_est; theta_est; r1_store; r2_store];
prefit = z_ip1-y;

%% sensitivity matrix
%initialize
H = zeros(N1+N2+2,10);

%range sensitivity
% line_of_sight = (x_ip1_g_i(1:3).'-x_ip1_g_i(6:8).')./norm((x_ip1_g_i(1:3).'-x_ip1_g_i(6:8).'));
% H(1,:) = [line_of_sight 0 0 -line_of_sight 0 0];
diff_x = x_ip1_g_i(1:3)-x_ip1_g_i(6:8);
Rot_red = R_rot_ecef_enu(1:2,:);

H(1,1) = (dot(Rot_red(1,:),diff_x.')*Rot_red(1,1)+dot(Rot_red(2,:),diff_x.')*Rot_red(2,1))/norm(Rot_red*diff_x);
H(1,6) = -H(1,1);
H(1,2) = (dot(Rot_red(1,:),diff_x.')*Rot_red(1,2)+dot(Rot_red(2,:),diff_x.')*Rot_red(2,2))/norm(Rot_red*diff_x);
H(1,7) = -H(1,2);
H(1,3) = (dot(Rot_red(1,:),diff_x.')*Rot_red(1,3)+dot(Rot_red(2,:),diff_x.')*Rot_red(2,3))/norm(Rot_red*diff_x);
H(1,8) = -H(1,3);


%theta sensitivity
vec_temp = [1/(-range_est*R_rot(1,1)*sind(theta_est)+range_est*R_rot(1,2)*cosd(theta_est)) ...
    1/(-range_est*R_rot(2,1)*sind(theta_est)+range_est*R_rot(2,2)*cosd(theta_est)) ...
    1/(-range_est*R_rot(3,1)*sind(theta_est)+range_est*R_rot(3,2)*cosd(theta_est))];
H(2,:) = [vec_temp 0 0 -vec_temp 0 0];

%pseudorange sensitivity
%car 1
for ind=1:N1
    r1 = -(sat_pos_t1(:,ind)-x_ip1_g_i(1:3)).';
    H(ind+2,:) = [r1/norm(r1) 1 0 zeros(1,5)];
end
%car 2
for ind=1:N2
    r2 = -(sat_pos_t2(:,ind)-x_ip1_g_i(6:8)).';
    H(ind+N1+2,:) = [zeros(1,5) r2/norm(r2) 1 0];
end

%% Kalman Gain
K = P_ip1_g_i*H.'/(R+H*P_ip1_g_i*H.');

%% update x and P
x_ip1_g_ip1 = x_ip1_g_i+K*prefit;
P_ip1_g_ip1 = (eye(10)-K*H)*P_ip1_g_i*(eye(10)-K*H).'+K*R*K.';

    
