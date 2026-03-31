% COMPARE_SAME_CHILDREN  Apply corrected threshold to Python's exact children.
% This eliminates cursor-range differences — both test the same children.
%
% Usage: octave --no-gui baseline/compare_same_children.m

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = 'baseline';
end
addpath(script_dir);

fprintf('============================================================\n');
fprintf('  Same-children threshold comparison\n');
fprintf('  Octave CORRECTED threshold vs Python threshold decisions\n');
fprintf('============================================================\n\n');

configs = {
    10, 1.2, 1;
    10, 1.2, 2;
    10, 1.3, 1;
    10, 1.3, 2;
    10, 1.4, 1;
    10, 1.4, 2;
};

for cfg_idx = 1:size(configs, 1)
    m_val = configs{cfg_idx, 1};
    c_target = configs{cfg_idx, 2};
    level = configs{cfg_idx, 3};
    tag = round(c_target * 100);
    gridSpace = 1.0 / m_val;

    fname = fullfile(script_dir, sprintf('children_L%d_m%d_c%d.mat', level, m_val, tag));
    if ~exist(fname, 'file')
        fprintf('  SKIP: %s not found\n', fname);
        continue;
    end

    data = load(fname);
    children_w = data.children_weights;  % N x d_child, continuous weights
    py_survived = logical(data.python_survived(:));
    n_children = size(children_w, 1);
    d_child = size(children_w, 2);
    numBins = d_child;

    fprintf('--- m=%d, c=%.2f, L%d: %d children (d_child=%d) ---\n', ...
        m_val, c_target, level, n_children, d_child);

    % Precompute matrices for this d_child
    [pairs, sumIndicesStore, binsContribute] = precompute_matrices(numBins);

    % Compute pairwise products for ALL children at once
    functionMult = children_w(:, pairs(:,1)) .* children_w(:, pairs(:,2));

    % Run corrected threshold on all children
    oct_pruned = false(n_children, 1);

    for j = 2:2*numBins
        active = ~oct_pruned;
        if sum(active) == 0
            break;
        end

        convFunctionVals = functionMult(active, :) * sumIndicesStore{j};
        normFactor = (2*numBins) / j;
        convFunctionVals = convFunctionVals * normFactor;

        % CORRECTED threshold
        boundToBeat = c_target + normFactor * (gridSpace^2 + ...
            2 * gridSpace * (children_w(active, :) * binsContribute{j}));

        checkBins = sum(convFunctionVals >= boundToBeat, 2);
        newly_pruned = checkBins > 0;

        idx = find(active);
        oct_pruned(idx(newly_pruned)) = true;
    end

    oct_survived = ~oct_pruned;

    % Compare
    n_py_surv = sum(py_survived);
    n_oct_surv = sum(oct_survived);
    agree = sum(oct_survived == py_survived);
    disagree = n_children - agree;

    % Breakdown of disagreements
    py_yes_oct_no = sum(py_survived & ~oct_survived);  % Python keeps, Octave prunes
    py_no_oct_yes = sum(~py_survived & oct_survived);  % Python prunes, Octave keeps

    fprintf('  Python survived:    %d\n', n_py_surv);
    fprintf('  Octave survived:    %d\n', n_oct_surv);
    fprintf('  Agree:              %d / %d (%.4f%%)\n', agree, n_children, 100*agree/n_children);
    fprintf('  Disagree:           %d\n', disagree);
    if disagree > 0
        fprintf('    Py=surv, Oct=pruned: %d (Octave more aggressive)\n', py_yes_oct_no);
        fprintf('    Py=pruned, Oct=surv: %d (Python more aggressive)\n', py_no_oct_yes);
    end
    fprintf('\n');
end

fprintf('============================================================\n');
fprintf('  Done.\n');
fprintf('============================================================\n');
