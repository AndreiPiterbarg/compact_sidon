% RUN_OCTAVE_BENCHMARK  Benchmark the original CS14 algorithm in GNU Octave.
%
% Processes sampled L4 parent bins through the Cloninger-Steinerberger (2014)
% branch-and-prune algorithm (adapted for CPU, no GPU/parallel) and records
% per-parent wall-clock timing for direct comparison with our optimized kernel.
%
% Only L4 is benchmarked: L3 survivors (d=32, integer bins summing to m=20)
% map to continuous weights summing to 1.0, which the MATLAB algorithm requires.
% Earlier checkpoints have bins summing to m/2 and would need normalization
% adjustments incompatible with a faithful reproduction.
%
% Prerequisites:
%   1. Install GNU Octave (https://octave.org)
%   2. Run: python -m baseline.export_test_parents
%      to create baseline/test_parents_L4.mat
%
% Usage (from project root):
%   octave --no-gui baseline/run_octave_benchmark.m
%
% Output:
%   baseline/octave_results_L4.mat  — per-parent timing data

% --- Setup: add baseline/ to path for benchmark_cs14.m ---
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = 'baseline';
end
addpath(script_dir);

fprintf('============================================================\n');
fprintf('  CS14 Octave Benchmark\n');
fprintf('  Algorithm: Cloninger & Steinerberger (2014)\n');
fprintf('  Adapted from initial_baseline.m (no GPU, no parallel)\n');
fprintf('============================================================\n\n');
fprintf('  ESTIMATED TIMES:\n');
fprintf('    Pre-computation (d=64):  ~30-120 seconds\n');
fprintf('    Per L4 parent:           ~30-90 seconds\n');
fprintf('    50 parents total:        ~25-75 minutes\n');
fprintf('    (Heavily depends on CPU single-thread speed)\n\n');

% --- Configuration ---
level_name = 'L4';
d_parent = 32;
numBins = 2 * d_parent;  % d_child = 64
% Max children per batch — matches original: floor(600000*576 / numBins^2)
% Original MATLAB: memBuffer = 600000*576; availableMem = floor(memBuffer/numBins^2);
memBufferRows = floor(600000 * 576 / numBins^2);  % = 84375 for numBins=64

mat_file = fullfile(script_dir, ['test_parents_' level_name '.mat']);
if ~exist(mat_file, 'file')
    fprintf('ERROR: test data not found (%s)\n', mat_file);
    fprintf('Run first: python -m baseline.export_test_parents\n');
    return;
end

fprintf('--- %s: d = %d -> %d ---\n', level_name, d_parent, numBins);

% === Load test parents ===
data = load(mat_file);
parents = data.parents;           % n_sample x d_parent, continuous weights
n_parents = size(parents, 1);
lowerBound = double(data.c_target);
gridSpace  = double(data.gridSpace);
m_val      = double(data.m);

% Validate: continuous weights should sum to ~1.0
weight_sum = sum(parents(1, :));
fprintf('  Loaded %d parents (d_parent=%d, m=%d, c=%.2f)\n', ...
    n_parents, d_parent, m_val, lowerBound);
fprintf('  Weight sum check: %.4f (expect 1.0)\n', weight_sum);
if abs(weight_sum - 1.0) > 0.01
    fprintf('  WARNING: weights do not sum to 1.0! MATLAB algorithm may give wrong results.\n');
end

% === Pre-compute pair and indicator matrices ===
fprintf('  Pre-computing matrices for d_child=%d (%d pairs)...\n', ...
    numBins, numBins^2);
t_pre = tic;

% All (i,j) pairs for i,j in 1:numBins
[xtmp, ytmp] = meshgrid(1:numBins, 1:numBins);
pairs = [xtmp(:) ytmp(:)];
numPairs = size(pairs, 1);

% subsetBins: indicator matrix — which conv indices each pair maps to
% Pair (i,j) with sum s=i+j maps to conv positions s-1 and s (1-indexed)
pairSums = sum(pairs, 2);
subsetBins = full(sparse( ...
    [(1:numPairs)'; (1:numPairs)'], ...
    [pairSums - 1;  pairSums], ...
    ones(2*numPairs, 1), numPairs, 2*numBins));

% For each window size j: which pairs contribute, which bins participate
sumIndicesStore = cell(2*numBins, 1);
binsContribute  = cell(2*numBins, 1);

for j = 2:2*numBins
    numIntervals = 2*numBins - j + 1;

    % Toeplitz matrix: each row = j consecutive 1s at different offsets
    row_vec = [1; zeros(numIntervals-1, 1)];
    col_vec = [ones(1, j), zeros(1, 2*numBins - j)];
    convBinIntervals = toeplitz(row_vec, col_vec);

    % Pair contributes to interval k iff BOTH its conv positions fall inside
    sumIndicesStore{j} = single((subsetBins * convBinIntervals') == 2);

    % Which child bins participate in each interval
    binsContribute{j} = zeros(numBins, numIntervals, 'single');
    for kk = 1:numIntervals
        contributing = logical(sumIndicesStore{j}(:, kk));
        if any(contributing)
            bins_used = unique(pairs(contributing, 1));
            binsContribute{j}(bins_used, kk) = single(1);
        end
    end
end

precomp_time = toc(t_pre);
fprintf('  Pre-computation done in %.1fs\n', precomp_time);

% === Warmup run (first parent, discard timing) ===
fprintf('  Warmup run...');
benchmark_cs14(parents(1, :), numBins, lowerBound, gridSpace, ...
    pairs, sumIndicesStore, binsContribute, memBufferRows);
fprintf(' done\n');

% === Timed benchmark ===
per_parent_times     = zeros(n_parents, 1);
per_parent_survivors = zeros(n_parents, 1);
per_parent_children  = zeros(n_parents, 1);

fprintf('  Processing %d parents...\n', n_parents);
t_total = tic;

for i = 1:n_parents
    [ns, nc, et] = benchmark_cs14(parents(i, :), numBins, lowerBound, ...
        gridSpace, pairs, sumIndicesStore, binsContribute, memBufferRows);

    per_parent_times(i)     = et;
    per_parent_survivors(i) = ns;
    per_parent_children(i)  = nc;

    elapsed_so_far = toc(t_total);
    avg_per_parent = elapsed_so_far / i;
    eta_sec = avg_per_parent * (n_parents - i);
    eta_min = eta_sec / 60;
    fprintf('    [%d/%d] %d children, %d survivors, %.1fs  (avg %.1fs/parent, ETA %.1f min)\n', ...
        i, n_parents, nc, ns, et, avg_per_parent, eta_min);
end

total_elapsed = toc(t_total);

% === Report ===
fprintf('\n  %s Results:\n', level_name);
fprintf('    Total wall time:     %.1fs\n', total_elapsed);
fprintf('    Mean per parent:     %.3fs\n', mean(per_parent_times));
fprintf('    Median per parent:   %.3fs\n', median(per_parent_times));
fprintf('    Min / Max:           %.3fs / %.3fs\n', ...
    min(per_parent_times), max(per_parent_times));
fprintf('    Total children:      %d\n', sum(per_parent_children));
fprintf('    Total survivors:     %d\n', sum(per_parent_survivors));
if sum(per_parent_children) > 0
    fprintf('    Survivor rate:       %.4f%%\n', ...
        100 * sum(per_parent_survivors) / sum(per_parent_children));
end
fprintf('    Throughput:          %.4f parents/sec\n', n_parents / total_elapsed);

% === Save results ===
outfile = fullfile(script_dir, ['octave_results_' level_name '.mat']);
c_target    = lowerBound;
seed        = double(data.seed);
n_total     = double(data.n_total);
precomp_sec = precomp_time;

save(outfile, 'per_parent_times', 'per_parent_survivors', ...
    'per_parent_children', 'd_parent', 'm_val', 'c_target', ...
    'n_parents', 'total_elapsed', 'seed', 'n_total', 'gridSpace', ...
    'precomp_sec', '-v7');

fprintf('    Saved to %s\n\n', outfile);

fprintf('============================================================\n');
fprintf('  Benchmark complete.\n');
fprintf('  Next: python -m baseline.run_comparison --levels L4\n');
fprintf('============================================================\n');
