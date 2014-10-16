function [res,grid_x,grid_y] = evaluate_basis_function_on_grid(w_linear,loc_x,loc_y);
[grid_x,grid_y] = meshgrid([loc_x],[loc_y]);
[sz_m,sz_n] = size(grid_x);

res = w_linear(1)*grid_x + w_linear(2)*grid_y + w_linear(3);

