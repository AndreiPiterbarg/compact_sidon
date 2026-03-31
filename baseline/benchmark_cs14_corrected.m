function [n_survivors, n_children, elapsed_sec] = benchmark_cs14_corrected(bin, numBins, lowerBound, gridSpace, pairs, sumIndicesStore, binsContribute, memBufferRows)
% BENCHMARK_CS14_CORRECTED  Process one parent bin through the CS14 algorithm.
%
% CORRECTED version: the threshold correction terms (gridSpace^2 + 2*gridSpace*W)
% are now multiplied by (2*numBins)/j, matching the normalization applied to
% the convolution values. Previously these terms were constant across all
% window sizes, which under-corrected for small windows by a factor of
% (2*numBins)/j.
%
% Arguments:
%   bin              - 1 x d_parent vector of parent bin weights (continuous)
%   numBins          - number of child bins (d_child = 2 * d_parent)
%   lowerBound       - target lower bound c (e.g. 1.4)
%   gridSpace        - discretization step (1/m, e.g. 0.05)
%   pairs            - numBins^2 x 2 matrix of all index pairs (1-indexed)
%   sumIndicesStore  - cell{2:2*numBins} of indicator matrices (precomputed)
%   binsContribute   - cell{2:2*numBins} of bin contribution matrices (precomputed)
%   memBufferRows    - max children per batch (default: 100000)
%
% Returns:
%   n_survivors   - number of surviving child bins
%   n_children    - total Cartesian product size
%   elapsed_sec   - wall-clock processing time

tic;

if nargin < 8
    memBufferRows = floor(600000 * 576 / numBins^2);
end

sizeMatrix = numBins / 2;  % d_parent = length(bin)

% Max single-bin weight before trivial Cauchy-Schwarz violation
x = sqrt(lowerBound / numBins);

% === Generate sub-bin splits for each parent bin ===
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

% === Compute mixed-radix strides ===
numRepeats = cumprod(tmpLength);
numRepeats(2:end+1) = numRepeats;
numRepeats(1) = 1;
numRepeats = single(numRepeats);

numRows = prod(tmpLength(1:sizeMatrix));
n_children = numRows;

% === Batch processing (memory-limited) ===
iterateRows = 1:memBufferRows:numRows;
numCombos = length(iterateRows);
iterateRows(end+1) = numRows + 1;

total_survivors = 0;

for k = 1:numCombos
    % --- Generate batch of children via mixed-radix enumeration ---
    indexMatrix = single(iterateRows(k):iterateRows(k+1)-1);
    batchSize = length(indexMatrix);

    index = floor(bsxfun(@times, 1 ./ numRepeats(1:sizeMatrix), indexMatrix));
    index = bsxfun(@mod, index, tmpLength);
    offsets = [0; cumLength(1:sizeMatrix-1)];
    index = bsxfun(@plus, index, offsets) + 1;

    % Look up partition entries and reshape to (batchSize x numBins)
    matrix_tmp = tmpPartition(index(:), :)';
    matrix_tmp = reshape(matrix_tmp(:), [2*sizeMatrix, batchSize])';

    % --- Compute all pairwise products f_i * f_j ---
    functionMult = matrix_tmp(:, pairs(:,1)) .* matrix_tmp(:, pairs(:,2));

    aboveThreshold = zeros(batchSize, 1);
    indices = true(batchSize, 1);
    j = 2;
    stopCond = Inf;

    % --- Window scan: j = 2, 3, ..., 2*numBins ---
    while j <= 2*numBins && stopCond > 0
        % Dense matmul: sum products contributing to each interval of size j
        convFunctionVals = functionMult(indices, :) * sumIndicesStore{j};

        % Normalize by interval fraction of full support
        normFactor = (2*numBins) / j;
        convFunctionVals = convFunctionVals * normFactor;

        % Dynamic threshold: base + CORRECTED correction for mass in interval
        % The correction terms must be scaled by the same (2*numBins)/j factor
        % as the convolution values. Previously they were unscaled, which
        % under-corrected for windows smaller than the full support.
        boundToBeat = lowerBound + normFactor * (gridSpace^2 + ...
            2 * gridSpace * (matrix_tmp(indices, :) * binsContribute{j}));

        % Check if any interval exceeds threshold
        checkBins = sum(convFunctionVals >= boundToBeat, 2);

        % Mark children that exceeded as pruned
        aboveThreshold(indices) = aboveThreshold(indices) | checkBins;

        % Continue only with survivors
        indices = aboveThreshold == 0;
        stopCond = sum(indices);
        j = j + 1;
    end

    total_survivors = total_survivors + sum(aboveThreshold == 0);
end

n_survivors = total_survivors;
elapsed_sec = toc;

end
