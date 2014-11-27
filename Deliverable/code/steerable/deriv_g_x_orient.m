function res = deriv_g_x_orient(k,angle)
if k==1,
    res  = -2*cos(-angle);
else
    res  = -2*sin(-angle);
end

