"""Combined A/B benchmark for all three optimization ideas.

Generates parent data once and benchmarks each idea sequentially to avoid
CPU contention and redundant data generation.

For each idea:
  1. Verify correctness (identical survivors)
  2. Time baseline vs optimized on same workload
  3. Report speedup
"""
import sys
import os
import time
import math
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

import numba
from numba import njit

from cpu.run_cascade import (
    _fused_generate_and_prune_gray as _baseline_kernel,
    _compute_bin_ranges,
    run_level0,
    process_parent_fused,
)

M = 20
C_TARGET = 1.4


# =====================================================================
# Idea 1: Incremental prefix_c maintenance
# =====================================================================
@njit(cache=False)
def _kernel_idea1(parent_int, n_half_child, m, c_target,
                  lo_arr, hi_arr, out_buf):
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    assert m <= 200
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1
    J_MIN = 7
    n_subtree_pruned = 0
    partial_conv = np.empty(conv_len, dtype=np.int32)
    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0
    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]
    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i; nz_pos[i] = nz_count; nz_count += 1
    # IDEA1: init prefix_c once
    prefix_c[0] = 0
    for i in range(d_child):
        prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])
    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        dyn_base_ell_arr[ell - 2] = c_target * m_d * m_d * np.float64(ell) * inv_4n
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc+1,hc+2,hc+3,hc,hc-1,hc+4,hc+5,hc-2,hc+6,hc-3,hc+7,hc+8):
            if 2 <= ell <= 2*d_child and ell_used[ell-2]==0:
                ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
        for ell in (d_child,d_child+1,d_child-1,d_child+2,d_child-2,d_child*2,d_child+d_child//2):
            if 2 <= ell <= 2*d_child and ell_used[ell-2]==0:
                ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
    else:
        for ell in range(2, min(17, 2*d_child+1)):
            ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
        for ell in (d_child,d_child+1,d_child-1,d_child+2,d_child-2,d_child*2,d_child+d_child//2,d_child//2):
            if 2 <= ell <= 2*d_child and ell_used[ell-2]==0:
                ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
    for ell in range(2, 2*d_child+1):
        if ell_used[ell-2]==0:
            ell_order[oi]=np.int32(ell); oi+=1
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2*i] += ci*ci
            for j in range(i+1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i+j] += np.int32(2)*ci*cj
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent-1, -1, -1):
        r = hi_arr[i]-lo_arr[i]+1
        if r > 1:
            active_pos[n_active]=i; radix[n_active]=r; n_active+=1
    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active+1, dtype=np.int32)
    for i in range(n_active+1):
        gc_focus[i] = i
    while True:
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            dyn_x_qc = dyn_base_ell_arr[qc_ell-2] + 1.0 + eps_margin + 2.0*np.float64(qc_W_int)
            if ws_qc > np.int64(dyn_x_qc * one_minus_4eps):
                quick_killed = True
        if not quick_killed:
            # IDEA1: skip prefix_c recomputation — already maintained
            pruned = False
            for ell_oi in range(ell_count):
                if pruned: break
                ell = ell_order[ell_oi]; n_cv = ell-1; dyn_base_ell = dyn_base_ell_arr[ell-2]
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv): ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0: ws += np.int64(raw_conv[s_lo+n_cv-1]) - np.int64(raw_conv[s_lo-1])
                    lo_bin = s_lo-(d_child-1)
                    if lo_bin < 0: lo_bin = 0
                    hi_bin = s_lo+ell-2
                    if hi_bin > d_child-1: hi_bin = d_child-1
                    W_int = prefix_c[hi_bin+1]-prefix_c[lo_bin]
                    dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0*np.float64(W_int)
                    if ws > np.int64(dyn_x * one_minus_4eps):
                        pruned = True; qc_ell = np.int32(ell); qc_s = np.int32(s_lo); qc_W_int = W_int; break
            if not pruned:
                use_rev = False; half_d = d_child//2
                for i in range(half_d):
                    jj = d_child-1-i
                    if child[jj] < child[i]: use_rev = True; break
                    elif child[jj] > child[i]: break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child): out_buf[n_surv,i] = child[d_child-1-i]
                    else:
                        for i in range(d_child): out_buf[n_surv,i] = child[i]
                n_surv += 1
        j = gc_focus[0]
        if j == n_active: break
        gc_focus[0] = 0
        pos = active_pos[j]; gc_a[j] += gc_dir[j]; cursor[pos] = lo_arr[pos]+gc_a[j]
        if gc_a[j]==0 or gc_a[j]==radix[j]-1:
            gc_dir[j] = -gc_dir[j]; gc_focus[j] = gc_focus[j+1]; gc_focus[j+1] = j+1
        k1 = 2*pos; k2 = k1+1
        old1 = np.int32(child[k1]); old2 = np.int32(child[k2])
        child[k1] = cursor[pos]; child[k2] = parent_int[pos]-cursor[pos]
        new1 = np.int32(child[k1]); new2 = np.int32(child[k2])
        delta1 = new1-old1; delta2 = new2-old2
        raw_conv[2*k1] += new1*new1 - old1*old1
        raw_conv[2*k2] += new2*new2 - old2*old2
        raw_conv[k1+k2] += np.int32(2)*(new1*new2 - old1*old2)
        if use_sparse:
            if old1!=0 and new1==0:
                p=nz_pos[k1]; nz_count-=1; last=nz_list[nz_count]; nz_list[p]=last; nz_pos[last]=p; nz_pos[k1]=-1
            elif old1==0 and new1!=0:
                nz_list[nz_count]=k1; nz_pos[k1]=nz_count; nz_count+=1
            if old2!=0 and new2==0:
                p=nz_pos[k2]; nz_count-=1; last=nz_list[nz_count]; nz_list[p]=last; nz_pos[last]=p; nz_pos[k2]=-1
            elif old2==0 and new2!=0:
                nz_list[nz_count]=k2; nz_pos[k2]=nz_count; nz_count+=1
            for idx in range(nz_count):
                jj = nz_list[idx]
                if jj!=k1 and jj!=k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1+jj] += np.int32(2)*delta1*cj
                    raw_conv[k2+jj] += np.int32(2)*delta2*cj
        else:
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj!=0: raw_conv[k1+jj]+=np.int32(2)*delta1*cj; raw_conv[k2+jj]+=np.int32(2)*delta2*cj
            for jj in range(k2+1, d_child):
                cj = np.int32(child[jj])
                if cj!=0: raw_conv[k1+jj]+=np.int32(2)*delta1*cj; raw_conv[k2+jj]+=np.int32(2)*delta2*cj
        # IDEA1: O(1) prefix_c update
        prefix_c[k2] += np.int64(delta1)
        if qc_ell > 0:
            qc_lo = qc_s-(d_child-1)
            if qc_lo < 0: qc_lo = 0
            qc_hi = qc_s+qc_ell-2
            if qc_hi > d_child-1: qc_hi = d_child-1
            if qc_lo <= k1 <= qc_hi: qc_W_int += np.int64(delta1)
            if qc_lo <= k2 <= qc_hi: qc_W_int += np.int64(delta2)
        if j == J_MIN and n_active > J_MIN:
            fpb = active_pos[J_MIN-1]; fl = 2*fpb
            if fl >= 4:
                pcl = 2*fl-1
                for kk in range(pcl): partial_conv[kk]=np.int32(0)
                for ii in range(fl):
                    ci=np.int32(child[ii])
                    if ci!=0:
                        partial_conv[2*ii]+=ci*ci
                        for jj2 in range(ii+1,fl):
                            cj2=np.int32(child[jj2])
                            if cj2!=0: partial_conv[ii+jj2]+=np.int32(2)*ci*cj2
                for kk in range(1,pcl): partial_conv[kk]+=partial_conv[kk-1]
                sub_pc = np.empty(fl+1, dtype=np.int64); sub_pc[0]=0
                for ii in range(fl): sub_pc[ii+1]=sub_pc[ii]+np.int64(child[ii])
                fup = fpb; sp = False
                for ell_oi in range(ell_count):
                    if sp: break
                    ell=ell_order[ell_oi]; n_cv=ell-1; dbe=dyn_base_ell_arr[ell-2]
                    nwp = pcl-n_cv+1
                    if nwp <= 0: continue
                    for s_lo in range(nwp):
                        ws=np.int64(partial_conv[s_lo+n_cv-1])
                        if s_lo>0: ws-=np.int64(partial_conv[s_lo-1])
                        lb=s_lo-(d_child-1)
                        if lb<0: lb=0
                        hb=s_lo+ell-2
                        if hb>d_child-1: hb=d_child-1
                        fh=hb
                        if fh>fl-1: fh=fl-1
                        if fh>=lb:
                            lc=lb
                            if lc<0: lc=0
                            wif=sub_pc[fh+1]-sub_pc[lc]
                        else: wif=np.int64(0)
                        ulb=lb
                        if ulb<fl: ulb=fl
                        if ulb<=hb:
                            pl=ulb//2; ph=hb//2
                            if pl<fup: pl=fup
                            if ph>=d_parent: ph=d_parent-1
                            if pl<=ph: wiu=parent_prefix[ph+1]-parent_prefix[pl]
                            else: wiu=np.int64(0)
                        else: wiu=np.int64(0)
                        wm=wif+wiu
                        dx=dbe+1.0+eps_margin+2.0*np.float64(wm)
                        if ws>np.int64(dx*one_minus_4eps): sp=True; break
                if sp:
                    n_subtree_pruned+=1; nf=gc_focus[J_MIN]
                    for kk in range(J_MIN): gc_a[kk]=0; gc_dir[kk]=1; gc_focus[kk]=kk
                    gc_focus[0]=nf; gc_focus[J_MIN]=J_MIN
                    for kk in range(J_MIN):
                        p=active_pos[kk]; cursor[p]=lo_arr[p]; child[2*p]=lo_arr[p]; child[2*p+1]=parent_int[p]-lo_arr[p]
                    for kk in range(conv_len): raw_conv[kk]=np.int32(0)
                    for ii in range(d_child):
                        ci=np.int32(child[ii])
                        if ci!=0:
                            raw_conv[2*ii]+=ci*ci
                            for jj2 in range(ii+1,d_child):
                                cj2=np.int32(child[jj2])
                                if cj2!=0: raw_conv[ii+jj2]+=np.int32(2)*ci*cj2
                    if use_sparse:
                        nz_count=0
                        for ii in range(d_child):
                            if child[ii]!=0: nz_list[nz_count]=ii; nz_pos[ii]=nz_count; nz_count+=1
                            else: nz_pos[ii]=-1
                    # IDEA1: recompute prefix_c after subtree reset
                    prefix_c[0]=0
                    for ii in range(d_child): prefix_c[ii+1]=prefix_c[ii]+np.int64(child[ii])
                    if qc_ell > 0:
                        ql=qc_s-(d_child-1)
                        if ql<0: ql=0
                        qh=qc_s+qc_ell-2
                        if qh>d_child-1: qh=d_child-1
                        qc_W_int=np.int64(0)
                        for ii in range(ql,qh+1): qc_W_int+=np.int64(child[ii])
                    continue
    return n_surv, n_subtree_pruned


# =====================================================================
# Idea 2: Multi-depth subtree pruning at J_OUTER
# =====================================================================
@njit(cache=False)
def _kernel_idea2(parent_int, n_half_child, m, c_target,
                  lo_arr, hi_arr, out_buf):
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    assert m <= 200
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1
    J_MIN = 7
    n_subtree_pruned = 0
    partial_conv = np.empty(conv_len, dtype=np.int32)
    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0
    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]
    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count]=i; nz_pos[i]=nz_count; nz_count+=1
    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        dyn_base_ell_arr[ell - 2] = c_target * m_d * m_d * np.float64(ell) * inv_4n
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc+1,hc+2,hc+3,hc,hc-1,hc+4,hc+5,hc-2,hc+6,hc-3,hc+7,hc+8):
            if 2 <= ell <= 2*d_child and ell_used[ell-2]==0:
                ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
        for ell in (d_child,d_child+1,d_child-1,d_child+2,d_child-2,d_child*2,d_child+d_child//2):
            if 2 <= ell <= 2*d_child and ell_used[ell-2]==0:
                ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
    else:
        for ell in range(2, min(17, 2*d_child+1)):
            ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
        for ell in (d_child,d_child+1,d_child-1,d_child+2,d_child-2,d_child*2,d_child+d_child//2,d_child//2):
            if 2 <= ell <= 2*d_child and ell_used[ell-2]==0:
                ell_order[oi]=np.int32(ell); ell_used[ell-2]=np.int32(1); oi+=1
    for ell in range(2, 2*d_child+1):
        if ell_used[ell-2]==0:
            ell_order[oi]=np.int32(ell); oi+=1
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2*i] += ci*ci
            for j in range(i+1, d_child):
                cj = np.int32(child[j])
                if cj != 0: raw_conv[i+j] += np.int32(2)*ci*cj
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent-1, -1, -1):
        r = hi_arr[i]-lo_arr[i]+1
        if r > 1:
            active_pos[n_active]=i; radix[n_active]=r; n_active+=1
    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active+1, dtype=np.int32)
    for i in range(n_active+1): gc_focus[i]=i
    # IDEA2: compute J_OUTER
    J_OUTER = min(2*J_MIN, n_active-1) if n_active > J_MIN+1 else -1

    while True:
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell-1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s+n_cv_qc): ws_qc += np.int64(raw_conv[k])
            dyn_x_qc = dyn_base_ell_arr[qc_ell-2]+1.0+eps_margin+2.0*np.float64(qc_W_int)
            if ws_qc > np.int64(dyn_x_qc*one_minus_4eps): quick_killed = True
        if not quick_killed:
            prefix_c[0]=0
            for i in range(d_child): prefix_c[i+1]=prefix_c[i]+np.int64(child[i])
            pruned = False
            for ell_oi in range(ell_count):
                if pruned: break
                ell=ell_order[ell_oi]; n_cv=ell-1; dbe=dyn_base_ell_arr[ell-2]
                nw = conv_len-n_cv+1
                ws=np.int64(0)
                for k in range(n_cv): ws+=np.int64(raw_conv[k])
                for s_lo in range(nw):
                    if s_lo>0: ws+=np.int64(raw_conv[s_lo+n_cv-1])-np.int64(raw_conv[s_lo-1])
                    lb=s_lo-(d_child-1)
                    if lb<0: lb=0
                    hb=s_lo+ell-2
                    if hb>d_child-1: hb=d_child-1
                    W_int=prefix_c[hb+1]-prefix_c[lb]
                    dx=dbe+1.0+eps_margin+2.0*np.float64(W_int)
                    if ws>np.int64(dx*one_minus_4eps):
                        pruned=True; qc_ell=np.int32(ell); qc_s=np.int32(s_lo); qc_W_int=W_int; break
            if not pruned:
                use_rev=False; half_d=d_child//2
                for i in range(half_d):
                    jj=d_child-1-i
                    if child[jj]<child[i]: use_rev=True; break
                    elif child[jj]>child[i]: break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child): out_buf[n_surv,i]=child[d_child-1-i]
                    else:
                        for i in range(d_child): out_buf[n_surv,i]=child[i]
                n_surv+=1
        j=gc_focus[0]
        if j==n_active: break
        gc_focus[0]=0
        pos=active_pos[j]; gc_a[j]+=gc_dir[j]; cursor[pos]=lo_arr[pos]+gc_a[j]
        if gc_a[j]==0 or gc_a[j]==radix[j]-1:
            gc_dir[j]=-gc_dir[j]; gc_focus[j]=gc_focus[j+1]; gc_focus[j+1]=j+1
        k1=2*pos; k2=k1+1
        old1=np.int32(child[k1]); old2=np.int32(child[k2])
        child[k1]=cursor[pos]; child[k2]=parent_int[pos]-cursor[pos]
        new1=np.int32(child[k1]); new2=np.int32(child[k2])
        delta1=new1-old1; delta2=new2-old2
        raw_conv[2*k1]+=new1*new1-old1*old1
        raw_conv[2*k2]+=new2*new2-old2*old2
        raw_conv[k1+k2]+=np.int32(2)*(new1*new2-old1*old2)
        if use_sparse:
            if old1!=0 and new1==0:
                p=nz_pos[k1]; nz_count-=1; last=nz_list[nz_count]; nz_list[p]=last; nz_pos[last]=p; nz_pos[k1]=-1
            elif old1==0 and new1!=0:
                nz_list[nz_count]=k1; nz_pos[k1]=nz_count; nz_count+=1
            if old2!=0 and new2==0:
                p=nz_pos[k2]; nz_count-=1; last=nz_list[nz_count]; nz_list[p]=last; nz_pos[last]=p; nz_pos[k2]=-1
            elif old2==0 and new2!=0:
                nz_list[nz_count]=k2; nz_pos[k2]=nz_count; nz_count+=1
            for idx in range(nz_count):
                jj=nz_list[idx]
                if jj!=k1 and jj!=k2:
                    cj=np.int32(child[jj]); raw_conv[k1+jj]+=np.int32(2)*delta1*cj; raw_conv[k2+jj]+=np.int32(2)*delta2*cj
        else:
            for jj in range(k1):
                cj=np.int32(child[jj])
                if cj!=0: raw_conv[k1+jj]+=np.int32(2)*delta1*cj; raw_conv[k2+jj]+=np.int32(2)*delta2*cj
            for jj in range(k2+1, d_child):
                cj=np.int32(child[jj])
                if cj!=0: raw_conv[k1+jj]+=np.int32(2)*delta1*cj; raw_conv[k2+jj]+=np.int32(2)*delta2*cj
        if qc_ell > 0:
            qc_lo=qc_s-(d_child-1)
            if qc_lo<0: qc_lo=0
            qc_hi=qc_s+qc_ell-2
            if qc_hi>d_child-1: qc_hi=d_child-1
            if qc_lo<=k1<=qc_hi: qc_W_int+=np.int64(delta1)
            if qc_lo<=k2<=qc_hi: qc_W_int+=np.int64(delta2)

        # IDEA2: J_OUTER subtree pruning (before J_MIN)
        if J_OUTER > J_MIN and j == J_OUTER and n_active > J_OUTER:
            fpb_o = active_pos[J_OUTER-1]; fl_o = 2*fpb_o
            if fl_o >= 4:
                pcl_o = 2*fl_o-1
                for kk in range(pcl_o): partial_conv[kk]=np.int32(0)
                for ii in range(fl_o):
                    ci=np.int32(child[ii])
                    if ci!=0:
                        partial_conv[2*ii]+=ci*ci
                        for jj2 in range(ii+1,fl_o):
                            cj2=np.int32(child[jj2])
                            if cj2!=0: partial_conv[ii+jj2]+=np.int32(2)*ci*cj2
                for kk in range(1,pcl_o): partial_conv[kk]+=partial_conv[kk-1]
                spo=np.empty(fl_o+1,dtype=np.int64); spo[0]=0
                for ii in range(fl_o): spo[ii+1]=spo[ii]+np.int64(child[ii])
                fup_o=fpb_o; op=False
                for ell_oi in range(ell_count):
                    if op: break
                    ell=ell_order[ell_oi]; n_cv=ell-1; dbe=dyn_base_ell_arr[ell-2]
                    nwp=pcl_o-n_cv+1
                    if nwp<=0: continue
                    for s_lo in range(nwp):
                        ws=np.int64(partial_conv[s_lo+n_cv-1])
                        if s_lo>0: ws-=np.int64(partial_conv[s_lo-1])
                        lb=s_lo-(d_child-1); hb=s_lo+ell-2
                        if lb<0: lb=0
                        if hb>d_child-1: hb=d_child-1
                        fh=hb
                        if fh>fl_o-1: fh=fl_o-1
                        if fh>=lb:
                            lc=lb; wif=spo[fh+1]-spo[max(0,lc)]
                        else: wif=np.int64(0)
                        ulb=lb
                        if ulb<fl_o: ulb=fl_o
                        if ulb<=hb:
                            pl=ulb//2; ph=hb//2
                            if pl<fup_o: pl=fup_o
                            if ph>=d_parent: ph=d_parent-1
                            if pl<=ph: wiu=parent_prefix[ph+1]-parent_prefix[pl]
                            else: wiu=np.int64(0)
                        else: wiu=np.int64(0)
                        wm=wif+wiu; dx=dbe+1.0+eps_margin+2.0*np.float64(wm)
                        if ws>np.int64(dx*one_minus_4eps): op=True; break
                if op:
                    n_subtree_pruned+=1; nf=gc_focus[J_OUTER]
                    for kk in range(J_OUTER): gc_a[kk]=0; gc_dir[kk]=1; gc_focus[kk]=kk
                    gc_focus[0]=nf; gc_focus[J_OUTER]=J_OUTER
                    for kk in range(J_OUTER):
                        p=active_pos[kk]; cursor[p]=lo_arr[p]; child[2*p]=lo_arr[p]; child[2*p+1]=parent_int[p]-lo_arr[p]
                    for kk in range(conv_len): raw_conv[kk]=np.int32(0)
                    for ii in range(d_child):
                        ci=np.int32(child[ii])
                        if ci!=0:
                            raw_conv[2*ii]+=ci*ci
                            for jj2 in range(ii+1,d_child):
                                cj2=np.int32(child[jj2])
                                if cj2!=0: raw_conv[ii+jj2]+=np.int32(2)*ci*cj2
                    if use_sparse:
                        nz_count=0
                        for ii in range(d_child):
                            if child[ii]!=0: nz_list[nz_count]=ii; nz_pos[ii]=nz_count; nz_count+=1
                            else: nz_pos[ii]=-1
                    if qc_ell>0:
                        ql=qc_s-(d_child-1)
                        if ql<0: ql=0
                        qh=qc_s+qc_ell-2
                        if qh>d_child-1: qh=d_child-1
                        qc_W_int=np.int64(0)
                        for ii in range(ql,qh+1): qc_W_int+=np.int64(child[ii])
                    continue

        # Original J_MIN subtree pruning
        if j == J_MIN and n_active > J_MIN:
            fpb = active_pos[J_MIN-1]; fl = 2*fpb
            if fl >= 4:
                pcl = 2*fl-1
                for kk in range(pcl): partial_conv[kk]=np.int32(0)
                for ii in range(fl):
                    ci=np.int32(child[ii])
                    if ci!=0:
                        partial_conv[2*ii]+=ci*ci
                        for jj2 in range(ii+1,fl):
                            cj2=np.int32(child[jj2])
                            if cj2!=0: partial_conv[ii+jj2]+=np.int32(2)*ci*cj2
                for kk in range(1,pcl): partial_conv[kk]+=partial_conv[kk-1]
                prefix_c[0]=0
                for ii in range(fl): prefix_c[ii+1]=prefix_c[ii]+np.int64(child[ii])
                fup=fpb; sp=False
                for ell_oi in range(ell_count):
                    if sp: break
                    ell=ell_order[ell_oi]; n_cv=ell-1; dbe=dyn_base_ell_arr[ell-2]
                    nwp=pcl-n_cv+1
                    if nwp<=0: continue
                    for s_lo in range(nwp):
                        ws=np.int64(partial_conv[s_lo+n_cv-1])
                        if s_lo>0: ws-=np.int64(partial_conv[s_lo-1])
                        lb=s_lo-(d_child-1); hb=s_lo+ell-2
                        if lb<0: lb=0
                        if hb>d_child-1: hb=d_child-1
                        fh=hb
                        if fh>fl-1: fh=fl-1
                        if fh>=lb:
                            lc=lb; wif=prefix_c[fh+1]-prefix_c[max(0,lc)]
                        else: wif=np.int64(0)
                        ulb=lb
                        if ulb<fl: ulb=fl
                        if ulb<=hb:
                            pl=ulb//2; ph=hb//2
                            if pl<fup: pl=fup
                            if ph>=d_parent: ph=d_parent-1
                            if pl<=ph: wiu=parent_prefix[ph+1]-parent_prefix[pl]
                            else: wiu=np.int64(0)
                        else: wiu=np.int64(0)
                        wm=wif+wiu; dx=dbe+1.0+eps_margin+2.0*np.float64(wm)
                        if ws>np.int64(dx*one_minus_4eps): sp=True; break
                if sp:
                    n_subtree_pruned+=1; nf=gc_focus[J_MIN]
                    for kk in range(J_MIN): gc_a[kk]=0; gc_dir[kk]=1; gc_focus[kk]=kk
                    gc_focus[0]=nf; gc_focus[J_MIN]=J_MIN
                    for kk in range(J_MIN):
                        p=active_pos[kk]; cursor[p]=lo_arr[p]; child[2*p]=lo_arr[p]; child[2*p+1]=parent_int[p]-lo_arr[p]
                    for kk in range(conv_len): raw_conv[kk]=np.int32(0)
                    for ii in range(d_child):
                        ci=np.int32(child[ii])
                        if ci!=0:
                            raw_conv[2*ii]+=ci*ci
                            for jj2 in range(ii+1,d_child):
                                cj2=np.int32(child[jj2])
                                if cj2!=0: raw_conv[ii+jj2]+=np.int32(2)*ci*cj2
                    if use_sparse:
                        nz_count=0
                        for ii in range(d_child):
                            if child[ii]!=0: nz_list[nz_count]=ii; nz_pos[ii]=nz_count; nz_count+=1
                            else: nz_pos[ii]=-1
                    if qc_ell>0:
                        ql=qc_s-(d_child-1)
                        if ql<0: ql=0
                        qh=qc_s+qc_ell-2
                        if qh>d_child-1: qh=d_child-1
                        qc_W_int=np.int64(0)
                        for ii in range(ql,qh+1): qc_W_int+=np.int64(child[ii])
                    continue
    return n_surv, n_subtree_pruned


# =====================================================================
# Benchmark harness
# =====================================================================
def generate_parents():
    """Generate L1 and L2 survivors for benchmarking."""
    print("Generating cascade data (shared across all benchmarks)...")
    l0 = run_level0(n_half=2, m=M, c_target=C_TARGET, verbose=False)
    l0_surv = l0['survivors']
    print(f"  L0: {len(l0_surv)} survivors")

    all_l1 = []
    for parent in l0_surv:
        surv, tc = process_parent_fused(parent, M, C_TARGET, 4)
        if len(surv) > 0:
            all_l1.append(surv)
    if all_l1:
        from cpu.run_cascade import _fast_dedup, _canonicalize_inplace
        l1_surv = np.vstack(all_l1)
        _canonicalize_inplace(l1_surv)
        l1_surv = _fast_dedup(l1_surv)
    else:
        l1_surv = np.empty((0, 8), dtype=np.int32)
    print(f"  L1: {len(l1_surv)} survivors")

    # Generate L2 from a sample of L1 (cap at 5000 to keep generation fast)
    sample_size = min(5000, len(l1_surv))
    rng = np.random.RandomState(42)
    l1_sample = l1_surv[rng.choice(len(l1_surv), sample_size, replace=False)]

    all_l2 = []
    for parent in l1_sample:
        surv, tc = process_parent_fused(parent, M, C_TARGET, 8)
        if len(surv) > 0:
            all_l2.append(surv)
    if all_l2:
        from cpu.run_cascade import _fast_dedup, _canonicalize_inplace
        l2_surv = np.vstack(all_l2)
        _canonicalize_inplace(l2_surv)
        l2_surv = _fast_dedup(l2_surv)
    else:
        l2_surv = np.empty((0, 16), dtype=np.int32)
    print(f"  L2: {len(l2_surv)} survivors (from {sample_size} L1 parents)")

    return l1_surv, l2_surv


def prepare_workload(parents, n_half_child, n_parents=200, min_children=10):
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    valid = []
    for i in range(len(parents)):
        result = _compute_bin_ranges(parents[i], M, C_TARGET,
                                     d_child, n_half_child)
        if result is not None:
            lo, hi, tc = result
            if tc >= min_children:
                valid.append((parents[i].copy(), lo, hi, tc))
    valid.sort(key=lambda x: -x[3])
    return valid[:n_parents]


def run_ab(workload, n_half_child, kernel_opt, label_opt, n_runs=5):
    """A/B comparison: baseline vs optimized kernel."""
    d_parent = workload[0][0].shape[0]
    d_child = 2 * d_parent
    total_children = sum(tc for _, _, _, tc in workload)

    print(f"\n  {label_opt}  ({len(workload)} parents, "
          f"{total_children:,} children)")

    # Warmup
    p0, lo0, hi0, tc0 = workload[0]
    buf = np.empty((min(tc0, 500_000), d_child), dtype=np.int32)
    _baseline_kernel(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)
    kernel_opt(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)

    # Quick correctness check (10 parents, count only)
    ok = True
    for parent, lo, hi, tc in workload[:10]:
        buf_cap = min(tc, 5_000_000)
        ba = np.empty((buf_cap, d_child), dtype=np.int32)
        bb = np.empty((buf_cap, d_child), dtype=np.int32)
        na, _ = _baseline_kernel(parent, n_half_child, M, C_TARGET, lo, hi, ba)
        nb, _ = kernel_opt(parent, n_half_child, M, C_TARGET, lo, hi, bb)
        if na != nb:
            print(f"    MISMATCH: baseline={na}, opt={nb}")
            ok = False
            break
    print(f"    Correctness: {'PASS' if ok else 'FAIL'}")
    if not ok:
        return None

    # Interleaved timing (reduces systematic bias from thermal throttling)
    baseline_times = []
    opt_times = []
    for run in range(n_runs):
        # Baseline
        t0 = time.perf_counter()
        for parent, lo, hi, tc in workload:
            buf = np.empty((min(tc, 5_000_000), d_child), dtype=np.int32)
            _baseline_kernel(parent, n_half_child, M, C_TARGET, lo, hi, buf)
        baseline_times.append(time.perf_counter() - t0)

        # Optimized
        t0 = time.perf_counter()
        for parent, lo, hi, tc in workload:
            buf = np.empty((min(tc, 5_000_000), d_child), dtype=np.int32)
            kernel_opt(parent, n_half_child, M, C_TARGET, lo, hi, buf)
        opt_times.append(time.perf_counter() - t0)

    baseline_times.sort()
    opt_times.sort()
    b_med = baseline_times[len(baseline_times) // 2]
    o_med = opt_times[len(opt_times) // 2]
    speedup = b_med / o_med if o_med > 0 else float('inf')
    delta_pct = (1.0 - o_med / b_med) * 100 if b_med > 0 else 0

    b_rate = total_children / b_med / 1e6
    o_rate = total_children / o_med / 1e6

    print(f"    BASELINE: {b_med:.4f}s ({b_rate:.2f}M ch/s)")
    print(f"    OPTIMIZED: {o_med:.4f}s ({o_rate:.2f}M ch/s)")
    print(f"    >>> SPEEDUP: {speedup:.3f}x  ({delta_pct:+.1f}%)")
    return {'speedup': speedup, 'delta_pct': delta_pct,
            'baseline_s': b_med, 'opt_s': o_med}


def main():
    print("=" * 70)
    print("Combined Benchmark: Ideas 1, 2, 3")
    print("=" * 70)

    l1_surv, l2_surv = generate_parents()

    # Prepare workloads
    wl_l2 = prepare_workload(l1_surv, 8, n_parents=200)
    wl_l3 = prepare_workload(l2_surv, 16, n_parents=100,
                             min_children=100) if len(l2_surv) > 0 else []

    print("\nJIT compiling all kernels (one-time cost)...")
    if wl_l2:
        p0, lo0, hi0, tc0 = wl_l2[0]
        d_child = 2 * p0.shape[0]
        buf = np.empty((min(tc0, 100_000), d_child), dtype=np.int32)
        nhc = 8
        _baseline_kernel(p0, nhc, M, C_TARGET, lo0, hi0, buf)
        _kernel_idea1(p0, nhc, M, C_TARGET, lo0, hi0, buf)
        _kernel_idea2(p0, nhc, M, C_TARGET, lo0, hi0, buf)
    if wl_l3:
        p0, lo0, hi0, tc0 = wl_l3[0]
        d_child = 2 * p0.shape[0]
        buf = np.empty((min(tc0, 100_000), d_child), dtype=np.int32)
        nhc = 16
        _baseline_kernel(p0, nhc, M, C_TARGET, lo0, hi0, buf)
        _kernel_idea1(p0, nhc, M, C_TARGET, lo0, hi0, buf)
        _kernel_idea2(p0, nhc, M, C_TARGET, lo0, hi0, buf)
    print("JIT done.\n")

    results = {}

    # ---- L2 benchmarks (d_child=16) ----
    if wl_l2:
        print("=" * 70)
        print(f"L1 -> L2  (d_child=16)")
        print("=" * 70)

        r = run_ab(wl_l2, 8, _kernel_idea1, "Idea 1: Incremental prefix_c")
        if r: results['idea1_L2'] = r

        r = run_ab(wl_l2, 8, _kernel_idea2, "Idea 2: Multi-depth subtree")
        if r: results['idea2_L2'] = r

    # ---- L3 benchmarks (d_child=32) ----
    if wl_l3:
        print("\n" + "=" * 70)
        print(f"L2 -> L3  (d_child=32)")
        print("=" * 70)

        r = run_ab(wl_l3, 16, _kernel_idea1, "Idea 1: Incremental prefix_c")
        if r: results['idea1_L3'] = r

        r = run_ab(wl_l3, 16, _kernel_idea2, "Idea 2: Multi-depth subtree")
        if r: results['idea2_L3'] = r
    else:
        print("\n  (No L3 workload — skipping L3 benchmarks)")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, r in sorted(results.items()):
        print(f"  {key:20s}  {r['speedup']:.3f}x  ({r['delta_pct']:+.1f}%)  "
              f"base={r['baseline_s']:.4f}s  opt={r['opt_s']:.4f}s")


if __name__ == '__main__':
    main()
