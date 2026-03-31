% COMPARE_BUGGY_CORRECTED  Run both buggy and corrected CS14 thresholds
% side by side on L0->L1 and compare survivor counts.
%
% Usage: octave --no-gui baseline/compare_buggy_corrected.m

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = 'baseline';
end
addpath(script_dir);

fprintf('============================================================\n');
fprintf('  Buggy vs Corrected Threshold Comparison (Octave)\n');
fprintf('============================================================\n\n');

configs = {
    10, 1.2;
    10, 1.3;
    10, 1.4;
    20, 1.2;
    20, 1.3;
    20, 1.4;
};

for cfg_idx = 1:size(configs, 1)
    m_val = configs{cfg_idx, 1};
    c_target = configs{cfg_idx, 2};
    gridSpace = 1.0 / m_val;
    n_half = 2;
    d_parent = 2 * n_half;
    numBins = 2 * d_parent;

    fprintf('--- m=%d, c_target=%.2f, d_parent=%d -> d_child=%d ---\n', ...
        m_val, c_target, d_parent, numBins);

    L0 = generate_compositions(m_val, d_parent);
    fprintf('  L0 compositions: %d\n', size(L0, 1));

    L0_weights = double(L0) / m_val;

    [pairs, sumIndicesStore, binsContribute] = precompute_matrices(numBins);

    total_surv_buggy = 0;
    total_surv_corrected = 0;
    total_children = 0;

    for i = 1:size(L0_weights, 1)
        parent = L0_weights(i, :);

        [ns_b, nc] = prune_one_parent_buggy(parent, numBins, c_target, gridSpace, ...
            pairs, sumIndicesStore, binsContribute);
        [ns_c, ~] = prune_one_parent_corrected(parent, numBins, c_target, gridSpace, ...
            pairs, sumIndicesStore, binsContribute);

        total_surv_buggy = total_surv_buggy + ns_b;
        total_surv_corrected = total_surv_corrected + ns_c;
        total_children = total_children + nc;
    end

    fprintf('  Total children: %d\n', total_children);
    fprintf('  BUGGY survivors:     %d\n', total_surv_buggy);
    fprintf('  CORRECTED survivors: %d\n', total_surv_corrected);
    if total_surv_corrected > total_surv_buggy
        fprintf('  -> Corrected has %d MORE survivors (buggy over-prunes)\n', ...
            total_surv_corrected - total_surv_buggy);
    elseif total_surv_corrected < total_surv_buggy
        fprintf('  -> Corrected has %d FEWER survivors\n', ...
            total_surv_buggy - total_surv_corrected);
    else
        fprintf('  -> IDENTICAL\n');
    end
    fprintf('\n');
end

fprintf('============================================================\n');
fprintf('  Done. Compare CORRECTED column against Python run_cascade.\n');
fprintf('============================================================\n');
