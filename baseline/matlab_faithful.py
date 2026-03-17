"""MATLAB-faithful reimplementation of the CS14 algorithm in Python/NumPy.

Uses the same algorithmic approach as initial_baseline.m:
  - float64/float32 arithmetic (not integer)
  - Batch dense matrix multiplication for autoconvolution
  - Sequential window scan with progressive pruning
  - No incremental autoconvolution, no quick-check, no subtree pruning

This serves as a cross-check: if Python MATLAB-faithful timing ≈ Octave timing,
our Octave measurements are validated (both execute the same algorithm).

Usage:
    from baseline.matlab_faithful import CS14Faithful
    cs14 = CS14Faithful(d_child=64, c_target=1.4, m=20)
    n_surv, n_children, elapsed = cs14.process_parent(bin_weights)
"""
import math
import time

import numpy as np


class CS14Faithful:
    """MATLAB-faithful CS14 algorithm using dense batch matrix operations."""

    def __init__(self, d_child, c_target, m, mem_buffer_rows=100_000):
        self.d_child = d_child
        self.d_parent = d_child // 2
        self.c_target = c_target
        self.m = m
        self.gridSpace = 1.0 / m
        self.mem_buffer_rows = mem_buffer_rows
        self._precompute()

    def _precompute(self):
        """Pre-compute pair and indicator matrices (matches initial_baseline.m lines 88-122)."""
        numBins = self.d_child
        # All (i,j) pairs, 0-indexed
        ii, jj = np.meshgrid(np.arange(numBins), np.arange(numBins))
        self.pairs_0 = np.column_stack([ii.ravel(), jj.ravel()])
        numPairs = len(self.pairs_0)

        # Convert to 1-indexed for subsetBins (MATLAB convention)
        pairs_1 = self.pairs_0 + 1
        pair_sums = pairs_1[:, 0] + pairs_1[:, 1]  # range: 2 to 2*numBins

        # subsetBins: (numPairs x 2*numBins), 0-indexed columns
        # Pair (i,j) with 1-indexed sum s maps to 0-indexed columns s-2 and s-1
        subsetBins = np.zeros((numPairs, 2 * numBins), dtype=np.float32)
        rows = np.arange(numPairs)
        subsetBins[rows, pair_sums - 2] = 1.0
        subsetBins[rows, pair_sums - 1] = 1.0

        # Pre-compute indicator matrices for each window size j
        self.sumIndicesStore = {}
        self.binsContribute = {}

        for j in range(2, 2 * numBins + 1):
            numIntervals = 2 * numBins - j + 1

            # Toeplitz-like matrix: each row has j consecutive 1s
            convBinIntervals = np.zeros((numIntervals, 2 * numBins), dtype=np.float32)
            for k in range(numIntervals):
                convBinIntervals[k, k:k + j] = 1.0

            # Which pairs contribute fully to each interval
            indicator = subsetBins @ convBinIntervals.T  # (numPairs x numIntervals)
            self.sumIndicesStore[j] = (indicator == 2.0).astype(np.float32)

            # Which bins participate in each interval
            self.binsContribute[j] = np.zeros((numBins, numIntervals), dtype=np.float32)
            for k in range(numIntervals):
                contributing = np.where(self.sumIndicesStore[j][:, k] > 0)[0]
                if len(contributing) > 0:
                    bins_used = np.unique(self.pairs_0[contributing, 0])
                    self.binsContribute[j][bins_used, k] = 1.0

    def process_parent(self, bin_weights):
        """Process one parent bin through the CS14 algorithm.

        Parameters
        ----------
        bin_weights : (d_parent,) float array — continuous bin weights

        Returns
        -------
        n_survivors : int
        n_children  : int (Cartesian product size)
        elapsed_sec : float
        """
        t0 = time.perf_counter()

        numBins = self.d_child
        d_parent = self.d_parent
        gs = self.gridSpace
        lb = self.c_target

        # Max single-bin weight (Cauchy-Schwarz bound)
        x = math.sqrt(lb / numBins)

        # --- Generate sub-bin splits for each parent bin ---
        tmpPartition_list = []
        tmpLength = np.zeros(d_parent, dtype=np.int64)
        for j in range(d_parent):
            weight = float(bin_weights[j])
            startVal = round((weight - x) / gs) * gs
            endVal = round(min(weight, x) / gs) * gs
            lo = max(0.0, startVal)
            # arange with float step needs care; use linspace-like approach
            n_steps = max(0, int(round((endVal - lo) / gs))) + 1
            subBins = np.array([lo + k * gs for k in range(n_steps)], dtype=np.float64)
            if len(subBins) == 0:
                subBins = np.array([lo])
            partialBin = np.column_stack([subBins, np.maximum(weight - subBins, 0.0)])
            tmpPartition_list.append(partialBin.astype(np.float32))
            tmpLength[j] = len(subBins)

        tmpPartition = np.vstack(tmpPartition_list)  # (sum_choices x 2)
        cumLength = np.cumsum(tmpLength)

        # Mixed-radix strides: [1, L0, L0*L1, ...]
        numRepeats = np.ones(d_parent, dtype=np.float64)
        for i in range(1, d_parent):
            numRepeats[i] = numRepeats[i - 1] * tmpLength[i - 1]

        numRows = int(np.prod(tmpLength))
        n_children = numRows

        # --- Batch processing ---
        total_survivors = 0
        for batch_start in range(0, numRows, self.mem_buffer_rows):
            batch_end = min(batch_start + self.mem_buffer_rows, numRows)
            batchSize = batch_end - batch_start

            # Mixed-radix decomposition (1-based indexing as in MATLAB)
            indexMatrix = np.arange(batch_start + 1, batch_end + 1, dtype=np.float64)
            # index: (d_parent x batchSize)
            index = np.floor(np.outer(1.0 / numRepeats, indexMatrix)).astype(np.int64)
            index = index % tmpLength[:, None]
            offsets = np.zeros(d_parent, dtype=np.int64)
            offsets[1:] = cumLength[:-1]
            index = index + offsets[:, None]  # 0-based into tmpPartition

            # Reshape to (batchSize x numBins) — MATLAB interleaved layout:
            # [sub_0, comp_0, sub_1, comp_1, ...]
            # This matches the MATLAB reshape: tmpPartition(index(:),:)' ->
            # reshape((:), [2*sizeMatrix, batchSize])' which interleaves sub/comp.
            flat_idx = index.T.ravel()
            entries = tmpPartition[flat_idx, :]  # (batchSize * d_parent, 2)
            subs = entries[:, 0].reshape(batchSize, d_parent)
            comps = entries[:, 1].reshape(batchSize, d_parent)
            matrix_tmp = np.empty((batchSize, numBins), dtype=np.float32)
            matrix_tmp[:, 0::2] = subs   # even columns = sub-bin values
            matrix_tmp[:, 1::2] = comps  # odd columns = complement values

            # --- Pairwise products ---
            functionMult = (matrix_tmp[:, self.pairs_0[:, 0]] *
                            matrix_tmp[:, self.pairs_0[:, 1]])

            aboveThreshold = np.zeros(batchSize, dtype=bool)
            active = np.ones(batchSize, dtype=bool)

            for j in range(2, 2 * numBins + 1):
                if not np.any(active):
                    break

                # Dense matmul: autoconv test values for active children
                conv_vals = functionMult[active, :] @ self.sumIndicesStore[j]
                conv_vals = conv_vals * (2 * numBins) / j

                # Dynamic threshold
                bound = ((lb + gs * gs) +
                         2 * gs * (matrix_tmp[active, :] @ self.binsContribute[j]))

                check = np.sum(conv_vals >= bound, axis=1)

                active_idx = np.where(active)[0]
                newly_pruned = check > 0
                aboveThreshold[active_idx[newly_pruned]] = True
                active = ~aboveThreshold

            total_survivors += int(np.sum(~aboveThreshold))

        elapsed = time.perf_counter() - t0
        return total_survivors, n_children, elapsed
