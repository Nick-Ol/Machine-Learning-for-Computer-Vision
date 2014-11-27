function res = deriv_h_xx_orient(k,angle)
switch k
    case 1
        res = .9780*cos(angle)^3;
    case 2
        res = .9780*(-3)*cos(angle)^2*sin(angle);
    case 3
        res = .9780*(3)*cos(angle)*sin(angle)^2;
    case 4
        res = .9780*(-1)*sin(angle)^3;
end