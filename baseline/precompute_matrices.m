function [pairs, sumIndicesStore, binsContribute] = precompute_matrices(numBins)
    [xtmp, ytmp] = meshgrid(1:numBins, 1:numBins);
    pairs = [xtmp(:) ytmp(:)];
    numPairs = size(pairs, 1);

    pairSums = sum(pairs, 2);
    subsetBins = full(sparse( ...
        [(1:numPairs)'; (1:numPairs)'], ...
        [pairSums - 1;  pairSums], ...
        ones(2*numPairs, 1), numPairs, 2*numBins));

    sumIndicesStore = cell(2*numBins, 1);
    binsContribute  = cell(2*numBins, 1);

    for j = 2:2*numBins
        numIntervals = 2*numBins - j + 1;
        row_vec = [1; zeros(numIntervals-1, 1)];
        col_vec = [ones(1, j), zeros(1, 2*numBins - j)];
        convBinIntervals = toeplitz(row_vec, col_vec);

        sumIndicesStore{j} = single((subsetBins * convBinIntervals') == 2);

        binsContribute{j} = zeros(numBins, numIntervals, 'single');
        for kk = 1:numIntervals
            contributing = logical(sumIndicesStore{j}(:, kk));
            if any(contributing)
                bins_used = unique(pairs(contributing, 1));
                binsContribute{j}(bins_used, kk) = single(1);
            end
        end
    end
end
