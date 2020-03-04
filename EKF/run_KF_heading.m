clear all
close all

theta = 134;

opt_select_dist = 100; %[20, 50, or 100 yards];
opt_select_vel = 0; %[0 for stationary, 1 for moving];
init_filt
%init filt

% rotation rate of earth
omega_E = 7.292115000000000e-05;
%% true samsung   20   50    100
% lat             37.42983
% long           -122.159822



%% initialize
% var1 = 20^2*ones(size(var1));
% var2 = 20^2*ones(size(var2));
x_i_g_i = x_ip1_g_ip1;

P_i_g_i =  diag([60^2 60^2 60^2 50 10^2 60^2 60^2 60^2 50 10^2]);
Q = diag([0 0 0 10 5^2 0 0 0 10 5^2]);
R_pseudo = 20^2;
R_range = 2^2;
R_theta = 1^2;
x1_store = zeros(3,min(size(r1,1),size(r2,1))-1);
x2_store = x1_store;
l1_store = x1_store;
l2_store = l1_store;

for ind=2:min(size(r1,1),size(r2,1));
    rt_phone1 = r1(ind,:);
    rt_phone2 = r2(ind,:);

    
    %remove NaN
%     temp = rt_phone1+rt_phone2;
    notnan_id_r1 = ~isnan(rt_phone1);
    notnan_id_r2 = ~isnan(rt_phone2);
    rt_phone1 = rt_phone1(notnan_id_r1);
    rt_phone2 = rt_phone2(notnan_id_r2);
    sat_pos_t1 = (squeeze(sat_pos(ind,notnan_id_r1,1:3))).';
    sat_pos_t2 = (squeeze(sat_pos2(ind,notnan_id_r2,1:3))).';
%     inds = find(sum(sat_pos(ind,:,1:3)+sat_pos2(ind,:,1:3),3)~=0);
%     rt_phone1 = rt_phone1(inds);
%     rt_phone2 = rt_phone2(inds);
%     sat_pos_t = (squeeze(sat_pos(ind,inds,1:3))).';
    pseudoranges1 = [rt_phone1];
    pseudoranges2 = rt_phone2;
    
    %adjust R to match measurement size
    R = diag([R_range R_theta var1(notnan_id_r1) var2(notnan_id_r2)]);
%     R = diag([R_range R_pseudo*ones(1,length(rt_phone1)+length(rt_phone2))]);
    
    %call filter
    dt = 1;
    [x_ip1_g_ip1, P_ip1_g_ip1] = Stationary_KF_heading(x_i_g_i, P_i_g_i, pseudoranges1, pseudoranges2, sat_pos_t1, sat_pos_t2, range,Q,R,dt,theta);
    
    P(ind-1) = (trace(P_ip1_g_ip1));
    x_i_g_i=x_ip1_g_ip1;
    P_i_g_i=P_ip1_g_ip1;
    x1_store(:,ind-1) = x_i_g_i(1:3);
    x2_store(:,ind-1) = x_i_g_i(6:8);
    [l1_store(1,ind-1),l1_store(2,ind-1),l1_store(3,ind-1)]  = ECEF_to_LLA(x_i_g_i(1),x_i_g_i(2),x_i_g_i(3));
    [l2_store(1,ind-1),l2_store(2,ind-1),l2_store(3,ind-1)] = ECEF_to_LLA(x_i_g_i(6),x_i_g_i(7),x_i_g_i(8));
%     [llaDegDegM] = Xyz2Lla(x_i_g_i(1:3).');
%     l1_store2(1:3,ind-1) = llaDegDegM;
%     
%     [llaDegDegM] = Xyz2Lla(x_i_g_i(6:8).');
%     l2_store2(1:3,ind-1) = llaDegDegM;
end

figure()
plot(1:length(x1_store),x1_store(1,:),1:length(x1_store),x2_store(1,:))
ylabel('x')
xlabel('Time')

figure()
plot(1:length(x1_store),x1_store(2,:),1:length(x1_store),x2_store(2,:))
ylabel('y')
xlabel('Time')

figure()
plot(1:length(x1_store),x1_store(3,:),1:length(x1_store),x2_store(3,:))
ylabel('z')
xlabel('Time')

figure()
plot(l1_store(1,:),l1_store(2,:),l2_store(1,:),l2_store(2,:))
ylabel('Longitude')
xlabel('Latitude')

figure()
plot(1:length(x1_store),l1_store(3,:),1:length(x1_store),l2_store(3,:))
ylabel('height (m)')
xlabel('Time')
set(gca,'FontSize',20);

figure()
plot(1:length(l1_store(1,:)),l1_store(1,:),1:length(l2_store(1,:)),l2_store(1,:))
xlabel('Time')
ylabel('\phi (deg)')
set(gca,'FontSize',20);

figure()
plot(1:length(l1_store(1,:)),l1_store(2,:),1:length(l2_store(2,:)),l2_store(2,:))
xlabel('Time')
ylabel('\lambda (deg)')
set(gca,'FontSize',20);
