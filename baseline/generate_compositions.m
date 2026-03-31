function comps = generate_compositions(m, d)
    % Generate all non-negative integer compositions of m into d parts.
    if d == 1
        comps = m;
        return;
    end
    comps = [];
    for first = 0:m
        rest = generate_compositions(m - first, d - 1);
        n = size(rest, 1);
        comps = [comps; first * ones(n, 1), rest];
    end
end
