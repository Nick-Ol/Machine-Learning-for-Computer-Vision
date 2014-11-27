function res = deriv_g_xx_orient(k,angle)
switch k
    case 1
        res = .9213*cos(angle)^2;
    case 2
        res  = 1.843*(-2)*cos(angle)*sin(angle);
    case 3
        res = .9213*sin(angle)^2;
end

