% COMPARE_MULTILEVEL  Run corrected Octave on Python's parents for L1 and L2.
%
% Usage: octave --no-gui baseline/compare_multilevel.m

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = 'baseline';
end
addpath(script_dir);

fprintf('============================================================\n');
fprintf('  Multi-level Octave CORRECTED vs Python comparison\n');
fprintf('============================================================\n\n');

configs = {
    10, 1.2;
    10, 1.3;
    10, 1.4;
};

for cfg_idx = 1:size(configs, 1)
    m_val = configs{cfg_idx, 1};
    c_target = configs{cfg_idx, 2};
    gridSpace = 1.0 / m_val;
    tag = round(c_target * 100);

    fprintf('=== m=%d, c_target=%.2f ===\n', m_val, c_target);

    for level = 1:2
        mat_file = fullfile(script_dir, sprintf('test_L%d_m%d_c%d.mat', level, m_val, tag));
        if ~exist(mat_file, 'file')
            fprintf('  L%d: SKIP (file not found: %s)\n', level, mat_file);
            continue;
        end

        data = load(mat_file);
        parents = data.parents;
        n_parents = size(parents, 1);
        d_parent = size(parents, 2);
        numBins = 2 * d_parent;
        py_surv_raw = data.n_survivors_raw;
        py_surv_dedup = data.n_survivors_dedup;
        py_children = data.total_children;

        fprintf('  L%d: %d parents, d=%d->%d\n', level, n_parents, d_parent, numBins);

        [pairs, sumIndicesStore, binsContribute] = precompute_matrices(numBins);

        total_surv_corrected = 0;
        total_children = 0;
        t0 = clock();

        for i = 1:n_parents
            [ns_c, nc] = prune_one_parent_corrected(parents(i,:), numBins, c_target, ...
                gridSpace, pairs, sumIndicesStore, binsContribute);
            total_surv_corrected = total_surv_corrected + ns_c;
            total_children = total_children + nc;

            if mod(i, 500) == 0
                elapsed = etime(clock(), t0);
                eta = elapsed / i * (n_parents - i);
                fprintf('      [%d/%d] %.0fs elapsed, ETA %.0fs\n', i, n_parents, elapsed, eta);
            end
        end
        elapsed = etime(clock(), t0);

        fprintf('      Children:  Octave=%d  Python=%d', total_children, py_children);
        if total_children == py_children
            fprintf('  MATCH\n');
        else
            fprintf('  DIFF=%+d\n', total_children - py_children);
        end

        fprintf('      Survivors: Octave_corrected=%d  Python_raw=%d', ...
            total_surv_corrected, py_surv_raw);
        diff = total_surv_corrected - py_surv_raw;
        pct = 100 * abs(diff) / max(py_surv_raw, 1);
        fprintf('  diff=%+d (%.2f%%)\n', diff, pct);
        fprintf('      Time: %.1fs\n\n', elapsed);
    end
end

fprintf('============================================================\n');
fprintf('  Done.\n');
fprintf('============================================================\n');
