function [n_survivors, n_children] = prune_one_parent_buggy(bin, numBins, lowerBound, gridSpace, pairs, sumIndicesStore, binsContribute)
    sizeMatrix = numBins / 2;
    x = sqrt(lowerBound / numBins);

    tmpPartition = cell(sizeMatrix, 1);
    tmpLength = zeros(sizeMatrix, 1);
    for j = 1:sizeMatrix
        weight = bin(j);
        startVal = round((weight - x) / gridSpace) * gridSpace;
        endVal = round(min(weight, x) / gridSpace) * gridSpace;
        subBins = max(0, startVal) : gridSpace : endVal;
        partialBin = [subBins; max(weight - subBins, 0)]';
        tmpPartition{j} = single(partialBin);
        tmpLength(j) = length(subBins);
    end
    tmpPartition = cell2mat(tmpPartition);
    cumLength = cumsum(tmpLength);

    numRepeats = cumprod(tmpLength);
    numRepeats(2:end+1) = numRepeats;
    numRepeats(1) = 1;
    numRepeats = single(numRepeats);
    numRows = prod(tmpLength(1:sizeMatrix));
    n_children = numRows;

    if numRows == 0
        n_survivors = 0;
        return;
    end

    indexMatrix = single(1:numRows);
    batchSize = numRows;

    index = floor(bsxfun(@times, 1 ./ numRepeats(1:sizeMatrix), indexMatrix));
    index = bsxfun(@mod, index, tmpLength);
    offsets = [0; cumLength(1:sizeMatrix-1)];
    index = bsxfun(@plus, index, offsets) + 1;

    matrix_tmp = tmpPartition(index(:), :)';
    matrix_tmp = reshape(matrix_tmp(:), [2*sizeMatrix, batchSize])';

    functionMult = matrix_tmp(:, pairs(:,1)) .* matrix_tmp(:, pairs(:,2));

    aboveThreshold = zeros(batchSize, 1);
    indices = true(batchSize, 1);
    j = 2;
    stopCond = Inf;

    while j <= 2*numBins && stopCond > 0
        convFunctionVals = functionMult(indices, :) * sumIndicesStore{j};
        convFunctionVals = convFunctionVals * (2*numBins) / j;

        % BUGGY: correction NOT scaled by normFactor
        boundToBeat = (lowerBound + gridSpace^2) + ...
            2 * gridSpace * (matrix_tmp(indices, :) * binsContribute{j});

        checkBins = sum(convFunctionVals >= boundToBeat, 2);
        aboveThreshold(indices) = aboveThreshold(indices) | checkBins;
        indices = aboveThreshold == 0;
        stopCond = sum(indices);
        j = j + 1;
    end

    n_survivors = sum(aboveThreshold == 0);
end
