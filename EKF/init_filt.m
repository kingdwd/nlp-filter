%% init filt

%% load data
if opt_select_dist == 20 && opt_select_vel == 0;
    range = 20*0.9144; %yard to m
    
    sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_00_40onyxsatposecef.mat');
    range_phone1 = load('gnss_log_2020_02_05_09_00_40onyxranges.mat');
    
    sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_08_58_10satposecef.mat');
    range_phone2 =load('gnss_log_2020_02_05_08_58_10ranges.mat');   
elseif opt_select_dist == 50 && opt_select_vel == 0;
    range = 50*0.9144; %yard to m
    
    sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_09_49onyxsatposecef.mat');
    range_phone1 = load('gnss_log_2020_02_05_09_09_49onyxranges.mat');
    
    sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_09_07_20satposecef.mat');
    range_phone2 =load('gnss_log_2020_02_05_09_07_20ranges.mat');  
elseif opt_select_dist == 100 && opt_select_vel == 0;
    range = 100*0.9144; %yard to m
    
    sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_14_15onyxsatposecef.mat');
    range_phone1 = load('gnss_log_2020_02_05_09_14_15onyxranges.mat');
    
    sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_09_11_46satposecef.mat');
    range_phone2 =load('gnss_log_2020_02_05_09_11_46ranges.mat');  
elseif opt_select_dist == 20 && opt_select_vel == 1;
    range = 20*0.9144; %yard to m
    
    sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_19_39onyxsatposecef.mat');
    range_phone1 = load('gnss_log_2020_02_05_09_19_39onyxranges.mat');
    
    sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_09_17_05satposecef.mat');
    range_phone2 =load('gnss_log_2020_02_05_09_17_05ranges.mat');  
elseif opt_select_dist == 50 && opt_select_vel == 1;
    range = 50*0.9144; %yard to m
    
    sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_22_49onyxsatposecef.mat');
    range_phone1 = load('gnss_log_2020_02_05_09_22_49onyxranges.mat');
    
    sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_09_20_20satposecef.mat');
    range_phone2 =load('gnss_log_2020_02_05_09_20_20ranges.mat');  
elseif opt_select_dist == 100 && opt_select_vel == 1;
    range = 100*0.9144; %yard to m
    
    sat_pos_ECEF_phone1 = load('gnss_log_2020_02_05_09_25_30onyxsatposecef.mat');
    range_phone1 = load('gnss_log_2020_02_05_09_25_30onyxranges.mat');
    
    sat_pos_ECEF_phone2 =load('gnss_log_2020_02_05_09_22_59satposecef.mat');
    range_phone2 =load('gnss_log_2020_02_05_09_22_59ranges.mat');  
    
end
%% pull data
sat_pos = sat_pos_ECEF_phone1.svPoss;
sat_pos2 = sat_pos_ECEF_phone2.svPoss;

r1 = squeeze(range_phone1.pseudoranges(:,:,1))+squeeze(sat_pos(1:end,:,4))+3e8*squeeze(sat_pos(1:end,:,5));
prangerate = squeeze(range_phone1.pseudoranges(:,:,2));
sat_vel = range_phone1.pseudoranges(:,:,3:5);

r2 = squeeze(range_phone2.pseudoranges(:,:,1))+squeeze(sat_pos2(1:end,:,4))+3e8*squeeze(sat_pos2(1:end,:,5));
prangerate2 = squeeze(range_phone2.pseudoranges(:,:,2));
sat_vel2 = range_phone2.pseudoranges(:,:,3:5);

%% initialize position
x_ip1_g_ip1 = zeros(10,1);
% initialization
ls_tot
x_ip1_g_ip1(1:4) = x1;
x_ip1_g_ip1(6:9) = x2;

%% initialize velocity


v1 = zeros(3,size(sat_pos,1));
for ind=2:size(r1,1)
    for ind_sat = 1:size(sat_pos,2)
    inds_in = ~isnan(r1(ind,:));
    [temp, bratet] = least_squares_vel(sat_vel(ind,inds_in,:), sat_pos(ind,inds_in,1:3),prangerate(ind,inds_in), x_ip1_g_ip1(1:3));
    b0rate1(ind) = bratet;
    [v1(:,ind)] = temp(1:3);
    end
end
x_ip1_g_ip1(5) = mean(b0rate1);

for ind=2:size(r1,1)
    for ind_sat = 1:size(sat_pos,2)
%     temp = r1(3:end,ind)-r1(2:(end-1),ind);
%     diff(ind)=max(temp)-min(temp);
    if ~isnan(r1(ind, ind_sat))
%         diff1(ind-1,ind_sat) = r1(ind, ind_sat)-(x_ip1_g_ip1(4)+x_ip1_g_ip1(5)*(ind-1))-norm(squeeze(sat_pos(ind,ind_sat,1:3))-x_ip1_g_ip1(1:3));
          diff1(ind-1,ind_sat) = r1(ind, ind_sat)-(x_ls1(4,ind-1))-norm(squeeze(sat_pos(ind,ind_sat,1:3))-x_ls1(1:3,ind-1));
    else
        diff1(ind-1,ind_sat) = nan;
    end
    
    end
end
var1 = nanvar(diff1,[],1);




v2 = zeros(3,size(sat_pos2,1));

for ind=2:size(r2,1)
    
    
    inds_in = ~isnan(r2(ind,:));
    [temp, bratet]= least_squares_vel(sat_vel2(ind,inds_in,:), sat_pos2(ind,inds_in,1:3),prangerate2(ind,inds_in), x_ip1_g_ip1(6:8));
    b0rate2(ind) = bratet;
    [v2(:,ind)] = temp(1:3);

end
x_ip1_g_ip1(10) = mean(b0rate2);

for ind=2:size(r2,1)
    for ind_sat = 1:size(sat_pos2,2)
    if ~isnan(r2(ind, ind_sat))
%         diff2(ind-1,ind_sat) = r2(ind, ind_sat)-(x_ip1_g_ip1(9)+x_ip1_g_ip1(10)*(ind-1))-norm(squeeze(sat_pos2(ind,ind_sat,1:3))-x_ip1_g_ip1(6:8));
          diff2(ind-1,ind_sat) = r2(ind, ind_sat)-(x_ls2(4,ind-1))-norm(squeeze(sat_pos2(ind,ind_sat,1:3))-x_ls2(1:3,ind-1));

    else
        diff2(ind-1,ind_sat) = nan;
    end
    
    
    end
end
var2 = nanvar(diff2,[],1);


