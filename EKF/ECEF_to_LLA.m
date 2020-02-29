function [lat, long, h] = ECEF_to_LLA(x,y,z)
%% necessary constants
%longitude unchanged
long = rad2deg(atan2(y,x));

%Ferrari's method from wikipedia
a = 6378137;
e = 8.1819190842622e-2;
b =  sqrt(a^2*(1-e^2));
r = sqrt(x^2+y^2);
eprimesquare = (a^2-b^2)/b^2;
F = 54*b^2*z^2;
G = r^2+(1-e^2)*z^2-e^2*(a^2-b^2);
c = e^4*F*r^2/G^3;
s = nthroot(1+c+sqrt(c^2+2*c),3);
P = F/(3*(s+1/s+1)^2*G^2);
Q = sqrt(1+2*e^4*P);
r0 = -P*e^2*r/(1+Q)+sqrt(0.5*a^2*(1+1/Q)-P*(1-e^2)*z^2/Q/(1+Q)-0.5*P*r^2);
U = sqrt((r-e^2*r0)^2+z^2);
V = sqrt((r-e^2*r0)^2+(1-e^2)*z^2);
z0 = b^2*z/a/V;
h = U*(1-b^2/a/V);
lat = rad2deg(atan((z+eprimesquare*z0)/r));




