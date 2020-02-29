% sat_pos = svPoss;


t = 200;
x = [x_ip1_g_ip1(1:3); 0];
sat_ind_vec =1:size(r1,2);
        for iter=1:8
            count =1;
            G=[];
            dp = [];
            ind= ~isnan(r1(2,:));
            for sat_ind=sat_ind_vec(ind)
                %calculate G
                sat_vec = squeeze(sat_pos(t,sat_ind, 1:3));
                G(count,1:4) = [-((sat_vec-x(1:3)).')./norm(sat_vec-x(1:3)) 1];
                
                %calculate dp
                measurement = r1(t,sat_ind)+sat_pos(t,sat_ind,4)+sat_pos(t, sat_ind,5);
%                 measurement = r1(t,sat_ind);
                dp(count,1) = (measurement-norm(sat_vec-x(1:3)))-x(4);
                count=count+1;
            end
            %update x
%             dy = pinv(G)*dp;
%             dy = inv(G.'*G)*G.'*dp;
            dy = pinv(G)*dp;
            x = x+dy;
            fprintf('Part a estimate iteration number %.0f: [%f %f %f %f] \n', iter, x(1), x(2), x(3), x(4))
        end
x1=x;

clear dp
clear G
sat_ind_vec =1:size(r2,2);
x = [x_ip1_g_ip1(6:8); 0];
        for iter=1:8
            count =1;
            G=[];
            dp = [];
            ind= ~isnan(r1(2,:));
            for sat_ind=1:size(r2,2)
                %calculate G
                sat_vec = squeeze(sat_pos2(t,sat_ind, 1:3));
                G(sat_ind,1:4) = [-((sat_vec-x(1:3)).')./norm(sat_vec-x(1:3)) 1];
                
                %calculate dp
                measurement = r2(t,sat_ind);
                dp(sat_ind,1) = (measurement-norm(sat_vec-x(1:3)))-x(4);
            end
            %update x
%             dy = pinv(G)*dp;
            dy = inv(G.'*G)*G.'*dp;
            x = x+dy;
            fprintf('Part a estimate iteration number %.0f: [%f %f %f %f] \n', iter, x(1), x(2), x(3), x(4))
        end
x2 = x;