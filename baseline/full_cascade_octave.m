% FULL_CASCADE_OCTAVE  Independent cascade L0->L1->L2->L3.
%
% Usage: octave --no-gui baseline/full_cascade_octave.m

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = 'baseline';
end
addpath(script_dir);

fprintf('============================================================\n');
fprintf('  Full Independent Cascade (Octave CORRECTED)\n');
fprintf('============================================================\n\n');

configs = {
    10, 1.1;
    10, 1.2;
};

for cfg_idx = 1:size(configs, 1)
    m_val = configs{cfg_idx, 1};
    c_target = configs{cfg_idx, 2};
    gridSpace = 1.0 / m_val;
    d0 = 4;

    fprintf('=== m=%d, c_target=%.2f ===\n', m_val, c_target);

    % L0: all compositions
    L0_int = generate_compositions(m_val, d0);
    current_weights = double(L0_int) / m_val;
    fprintf('  L0: %d compositions (no pruning applied)\n', size(L0_int, 1));

    for level = 1:3
        d_parent = size(current_weights, 2);
        numBins = 2 * d_parent;
        n_parents = size(current_weights, 1);

        fprintf('  L%d: %d parents, d=%d->%d\n', level, n_parents, d_parent, numBins);

        if n_parents == 0
            fprintf('      PROVEN at L%d (0 parents)\n', level);
            break;
        end
        if n_parents > 300000
            fprintf('      TOO MANY, stopping\n');
            break;
        end

        [pairs, sumIndicesStore, binsContribute] = precompute_matrices(numBins);

        total_children = 0;
        total_survivors = 0;
        all_surv = [];
        t0 = clock();

        for i = 1:n_parents
            [ns, nc, ~, sw] = prune_one_parent_corrected_collect( ...
                current_weights(i,:), numBins, c_target, gridSpace, ...
                pairs, sumIndicesStore, binsContribute);
            total_children = total_children + nc;
            total_survivors = total_survivors + ns;
            if ns > 0
                all_surv = [all_surv; sw];
            end

            if mod(i, max(1, floor(n_parents/5))) == 0
                elapsed = etime(clock(), t0);
                eta = elapsed / i * (n_parents - i);
                fprintf('      [%d/%d] %d surv, %.0fs, ETA %.0fs\n', ...
                    i, n_parents, total_survivors, elapsed, eta);
            end
        end

        elapsed = etime(clock(), t0);

        % Dedup
        if size(all_surv, 1) > 1
            [all_surv, ~] = unique(all_surv, 'rows');
        end
        n_dedup = size(all_surv, 1);

        fprintf('      %d children -> %d raw -> %d dedup [%.1fs]\n', ...
            total_children, total_survivors, n_dedup, elapsed);

        current_weights = all_surv;
    end
    fprintf('\n');
end

fprintf('============================================================\n');
fprintf('  Done.\n');
fprintf('============================================================\n');
