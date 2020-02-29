clear all
close all
%% load data
%check that svids match?
sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_09_49onyxsatposecef.mat');
sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_09_07_20satposecef.mat');

sat_pos = sat_pos_ECEF_phone1.svPoss;
sat_pos2 = sat_pos_ECEF_phone2.svPoss;
% sat_pos = sat_pos(:,2:4).';

range = 50*0.9144; %yard to m
range_phone1 = load('gnss_log_2020_02_05_09_09_49onyxranges.mat');
r1 = range_phone1.pseudoranges(1:end,:)+squeeze(sat_pos(1:end,:,4));


range_phone2 =load('gnss_log_2020_02_05_09_07_20ranges.mat');
r2 = range_phone2.pseudoranges(1:end,:)+squeeze(sat_pos2(1:end,:,4));



%% initialize
% x_i_g_i = [-2699.293e3 -4293.079e3 3855.401e3 -2699.287e3 -4293.066e3 3855.411e3].';
% x_i_g_i = [-2699.293e3 -4293.079e3 3855.401e3 -2699.305e3 -4293.053e3 3855.423e3].';
x_i_g_i = [-2700.586e3 -4293.8877e3 3855.539e3 -2700.596e3 -4293.9000e3 3855.541e3].';
% x_i_g_i = [-23598746.57 4499231.45 3855.1e3 -13598746.57 4499231.45 3855.1e3].';

P_i_g_i = 1^2*eye(6);
Q = eye(6);
R_pseudo = 200^2;
R_range = 2^2;
x1_store = zeros(3,size(r1,1)-1);
x2_store = x1_store;
l1_store = x1_store;
l2_store = l1_store;

for ind=2:size(r1,1);
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
    R = diag([R_range R_pseudo*ones(1,length(rt_phone1)+length(rt_phone2))]);
    
    %call filter
    [x_ip1_g_ip1, P_ip1_g_ip1] = Stationary_KF(x_i_g_i, P_i_g_i, pseudoranges1, pseudoranges2, sat_pos_t1, sat_pos_t2, range,Q,R);
    
    P(ind-1) = (trace(P_ip1_g_ip1));
    x_i_g_i=x_ip1_g_ip1;
    P_i_g_i=P_ip1_g_ip1;
    x1_store(:,ind-1) = x_i_g_i(1:3);
    x2_store(:,ind-1) = x_i_g_i(4:6);
    [l1_store(1,ind-1),l1_store(2,ind-1),l1_store(3,ind-1)]  = ECEF_to_LLA(x_i_g_i(1),x_i_g_i(2),x_i_g_i(3));
    [l2_store(1,ind-1),l2_store(2,ind-1),l2_store(3,ind-1)] = ECEF_to_LLA(x_i_g_i(4),x_i_g_i(5),x_i_g_i(6));
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

figure()
plot(1:length(l1_store(1,:)),l1_store(1,:),1:length(l2_store(1,:)),l2_store(1,:))
xlabel('Time')
ylabel('Lat')

figure()
plot(1:length(l1_store(1,:)),l1_store(2,:),1:length(l2_store(2,:)),l2_store(2,:))
xlabel('Time')
ylabel('Long')
