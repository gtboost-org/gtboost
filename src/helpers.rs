//! Pure utility helpers used throughout the model.
//!
//! - Bitvec helpers (`bitvec_new`, `bitvec_set`, `bitvec_test`):
//!   packed in-sample-mask bookkeeping for honest leaves and OOB tracking.
//! - Linear-system solvers (`solve_small_linear_system`, `solve_spd`,
//!   `solve_spd_with_scratch`, `solve_spd_cg_with_scratch`):
//!   used by leaf-linear / leaf-quadratic refits.
//! - Gradient transform (`transform_gradients_for_split`):
//!   rank/sign-criterion preprocessing for split finding.
//!
//! These functions are stateless (no `self`) — extracted from `model.rs`
//! to keep that file focused on the boosting algorithm.

// ── Bitvec helpers for packed in-sample masks ───────────────────────────────

#[inline]
pub(crate) fn bitvec_new(n: usize) -> Vec<u64> {
    vec![0u64; (n + 63) / 64]
}

#[inline]
pub(crate) fn bitvec_set(v: &mut [u64], idx: usize) {
    v[idx / 64] |= 1u64 << (idx % 64);
}

#[inline]
pub(crate) fn bitvec_test(v: &[u64], idx: usize) -> bool {
    (v[idx / 64] >> (idx % 64)) & 1 != 0
}

// ── Small linear systems (Gaussian elimination with partial pivoting) ───────

/// Solve `A x = b` for a small dense system via Gaussian elimination.
/// Returns `None` if the matrix is singular or non-finite.
pub(crate) fn solve_small_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 || a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > pivot_abs {
                pivot = row;
                pivot_abs = v;
            }
        }
        if pivot_abs <= 1e-18 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        let diag = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            a[row][col] = 0.0;
            for k in (col + 1)..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0f64; n];
    for row_rev in 0..n {
        let row = n - 1 - row_rev;
        let mut rhs = b[row];
        for k in (row + 1)..n {
            rhs -= a[row][k] * x[k];
        }
        x[row] = rhs / a[row][row];
        if !x[row].is_finite() {
            return None;
        }
    }
    Some(x)
}

// ── Symmetric Positive-Definite solvers (Cholesky / CG fallback) ────────────

/// Solve `A x = b` for small SPD matrix `A` (n ≤ 8) via Cholesky.
/// Falls back to zero vector if factorization fails (singular).
pub(crate) fn solve_spd(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut l = Vec::new();
    let mut y = Vec::new();
    let mut x = vec![0.0f64; n];
    if solve_spd_with_scratch(n, a, b, &mut l, &mut y, &mut x) {
        x
    } else {
        vec![0.0; n]
    }
}

/// Cholesky-factor SPD solve with caller-provided scratch buffers (avoids
/// reallocation in tight loops). Falls back to CG for n > 64.
#[inline]
pub(crate) fn solve_spd_with_scratch(
    n: usize,
    a: &[f64],
    b: &[f64],
    l: &mut Vec<f64>,
    y: &mut Vec<f64>,
    x: &mut Vec<f64>,
) -> bool {
    if n == 0 || a.len() < n * n || b.len() < n {
        return false;
    }
    if n > 64 {
        return solve_spd_cg_with_scratch(n, a, b, l, y, x);
    }
    l.resize(n * n, 0.0);
    y.resize(n, 0.0);
    x.resize(n, 0.0);
    l[..n * n].fill(0.0);
    y[..n].fill(0.0);
    x[..n].fill(0.0);

    // Cholesky: A = L * L^T
    for j in 0..n {
        let mut sum = 0.0f64;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = a[j * n + j] - sum;
        if diag <= 1e-30 {
            return false;
        }
        l[j * n + j] = diag.sqrt();
        for i in (j + 1)..n {
            let mut sum2 = 0.0f64;
            for k in 0..j {
                sum2 += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (a[i * n + j] - sum2) / l[j * n + j];
        }
    }
    // Forward substitution: L * y = b
    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }
    // Back substitution: L^T * x = y
    for i in (0..n).rev() {
        let mut sum = 0.0f64;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }
    true
}

/// Conjugate-gradient SPD solve for larger systems. Used as fallback from
/// `solve_spd_with_scratch` when n > 64 (Cholesky cost grows as n³).
#[inline]
pub(crate) fn solve_spd_cg_with_scratch(
    n: usize,
    a: &[f64],
    b: &[f64],
    scratch: &mut Vec<f64>,
    r: &mut Vec<f64>,
    x: &mut Vec<f64>,
) -> bool {
    scratch.resize(3 * n, 0.0);
    r.resize(n, 0.0);
    x.resize(n, 0.0);
    x[..n].fill(0.0);

    let (p, rest) = scratch[..3 * n].split_at_mut(n);
    let (z, ap) = rest.split_at_mut(n);
    let mut b_norm2 = 0.0f64;
    for i in 0..n {
        let bi = b[i];
        if !bi.is_finite() {
            return false;
        }
        r[i] = bi;
        let diag = a[i * n + i].abs().max(1e-18);
        z[i] = bi / diag;
        p[i] = z[i];
        ap[i] = 0.0;
        b_norm2 += bi * bi;
    }

    let mut rz = (0..n).map(|i| r[i] * z[i]).sum::<f64>();
    if !rz.is_finite() {
        return false;
    }
    if rz.abs() <= 1e-30 || b_norm2 <= 1e-30 {
        return true;
    }

    let tol2 = (1e-8 * 1e-8) * b_norm2.max(1.0);
    let max_iter = n.min(12);
    for _ in 0..max_iter {
        for i in 0..n {
            let row = &a[i * n..(i + 1) * n];
            let mut s = 0.0f64;
            for j in 0..n {
                s += row[j] * p[j];
            }
            ap[i] = s;
        }
        let pap = (0..n).map(|i| p[i] * ap[i]).sum::<f64>();
        if !(pap.is_finite() && pap > 1e-30) {
            return false;
        }
        let alpha = rz / pap;
        let mut r_norm2 = 0.0f64;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            r_norm2 += r[i] * r[i];
        }
        if !r_norm2.is_finite() {
            return false;
        }
        if r_norm2 <= tol2 {
            return true;
        }
        for i in 0..n {
            let diag = a[i * n + i].abs().max(1e-18);
            z[i] = r[i] / diag;
        }
        let rz_new = (0..n).map(|i| r[i] * z[i]).sum::<f64>();
        if !rz_new.is_finite() {
            return false;
        }
        let beta = rz_new / rz.max(1e-30);
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    true
}

// ── Gradient-transform pre-pass for non-standard split criteria ─────────────

/// Transform gradients for split-finding. Leaf values are refit later using
/// original g, h.
///
/// - `"rank"`:  assign dense rank 1..N to each gradient (ties get average rank).
///              Hessians set to 1.0 uniformly (ranks have unit weight).
/// - `"sign"`:  replace with sign(g) in {-1, 0, +1}. Hessians set to 1.0.
/// - other:     no-op (returns None).
///
/// Cost: O(N log N) for rank, O(N) for sign. Called once per round.
pub(crate) fn transform_gradients_for_split(
    gradients: &[f64],
    criterion: &str,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = gradients.len();
    if n == 0 {
        return None;
    }
    match criterion {
        "rank" => {
            // Sort indices by gradient value, then assign ranks with average-tie
            // handling. Center around 0 so the sum-squared gain formula behaves
            // as a Wilcoxon rank-sum statistic (ranks preserve order, so the
            // centered_rank has the same sign as the original gradient except
            // exactly at the median).
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                gradients[a]
                    .partial_cmp(&gradients[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut ranks = vec![0.0f64; n];
            let mut i = 0;
            while i < n {
                let mut j = i;
                while j + 1 < n && (gradients[order[j + 1]] - gradients[order[i]]).abs() < 1e-12 {
                    j += 1;
                }
                let avg_rank = (i + j) as f64 / 2.0 + 1.0;
                for k in i..=j {
                    ranks[order[k]] = avg_rank;
                }
                i = j + 1;
            }
            let mean_rank = (n as f64 + 1.0) / 2.0;
            for r in ranks.iter_mut() {
                *r -= mean_rank;
            }
            let hess_out = vec![1.0f64; n];
            Some((ranks, hess_out))
        }
        "sign" => {
            let g_out: Vec<f64> = gradients
                .iter()
                .map(|&g| {
                    if g > 1e-12 {
                        1.0
                    } else if g < -1e-12 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let hess_out = vec![1.0f64; n];
            Some((g_out, hess_out))
        }
        _ => None,
    }
}
