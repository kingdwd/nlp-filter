function [xout, brate] = least_squares_vel(sat_vel, sat_pos,prangerate,pos);

omega_E = 7.292115000000000e-05;

            count =1;
            G=[];
            dp = [];
            
            for sat_ind=1:size(sat_vel,2)
                %calculate G
                xsat = squeeze(sat_pos(1,sat_ind, 1:3));
                vsat = squeeze(sat_vel(1,sat_ind, 1:3));
                G(sat_ind,1:4) = [-((xsat-pos)).'/norm(xsat-pos) 1];
                
                %calculate dp
                measurement = prangerate(sat_ind);
                
%                 measurement = r1(t,sat_ind);
                dp(sat_ind,1) = (measurement-dot((vsat), ((xsat-pos))./norm(xsat-pos)));
                count=count+1;
            end
            %update x
%             dy = pinv(G)*dp;
%             dy = inv(G.'*G)*G.'*dp;
            dy = pinv(G)*dp;
           
          
xout = (dy(1:3));
brate = dy(4);   