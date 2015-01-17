function val = pairwise(Xp_1, Xp_2, Xr_1, Xr_2, sigma_1, sigma_2, mu_1, mu_2)

val =  1/(2*pi*sigma_1*sigma_2)*exp(-((Xp_1-Xr_1-mu_1).^2/(2*sigma_1^2) ...
    + (Xp_2-Xr_2-mu_2).^2./(2*sigma_2^2)));
