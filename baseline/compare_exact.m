% COMPARE_EXACT  Load Python's L0 survivors, run corrected Octave pruning on
% L0->L1, report total survivors (before dedup) for direct comparison.
%
% Usage: octave --no-gui baseline/compare_exact.m

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = 'baseline';
end
addpath(script_dir);

fprintf('============================================================\n');
fprintf('  Octave CORRECTED threshold on Python L0 survivors\n');
fprintf('============================================================\n\n');

configs = {
    10, 1.2, 'test_L0_m10_c120.mat';
    10, 1.3, 'test_L0_m10_c130.mat';
    10, 1.4, 'test_L0_m10_c140.mat';
    20, 1.2, 'test_L0_m20_c120.mat';
    20, 1.3, 'test_L0_m20_c130.mat';
    20, 1.4, 'test_L0_m20_c140.mat';
};

for cfg_idx = 1:size(configs, 1)
    m_val = configs{cfg_idx, 1};
    c_target = configs{cfg_idx, 2};
    mat_name = configs{cfg_idx, 3};
    gridSpace = 1.0 / m_val;

    mat_file = fullfile(script_dir, mat_name);
    if ~exist(mat_file, 'file')
        fprintf('  SKIP: %s not found\n', mat_file);
        continue;
    end

    data = load(mat_file);
    parents = data.parents;
    n_parents = size(parents, 1);
    d_parent = size(parents, 2);
    numBins = 2 * d_parent;

    fprintf('--- m=%d, c_target=%.2f, %d parents, d=%d->%d ---\n', ...
        m_val, c_target, n_parents, d_parent, numBins);

    [pairs, sumIndicesStore, binsContribute] = precompute_matrices(numBins);

    total_surv_buggy = 0;
    total_surv_corrected = 0;
    total_children = 0;

    for i = 1:n_parents
        parent = parents(i, :);

        [ns_b, nc] = prune_one_parent_buggy(parent, numBins, c_target, gridSpace, ...
            pairs, sumIndicesStore, binsContribute);
        [ns_c, ~] = prune_one_parent_corrected(parent, numBins, c_target, gridSpace, ...
            pairs, sumIndicesStore, binsContribute);

        total_surv_buggy = total_surv_buggy + ns_b;
        total_surv_corrected = total_surv_corrected + ns_c;
        total_children = total_children + nc;
    end

    fprintf('  Total children:        %d\n', total_children);
    fprintf('  BUGGY survivors:       %d\n', total_surv_buggy);
    fprintf('  CORRECTED survivors:   %d\n', total_surv_corrected);
    fprintf('\n');
end

fprintf('============================================================\n');
fprintf('  Compare CORRECTED above with Python \"before dedup\" column\n');
fprintf('============================================================\n');
