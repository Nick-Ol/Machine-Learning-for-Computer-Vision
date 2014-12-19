function proba = proba(X, Marg_Distrib)
    proba = 1;
    for i=1:5
        proba = proba * Marg_Distrib{i}(X(i));
    end
    
end

