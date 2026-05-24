use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

fn validate_prior(n_classes: usize, global_prior: &[f64]) -> PyResult<()> {
    if n_classes == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_classes must be positive",
        ));
    }
    if global_prior.len() != n_classes {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "global_prior length must equal n_classes",
        ));
    }
    Ok(())
}

fn prior_entropy(global_prior: &[f64]) -> f64 {
    -global_prior
        .iter()
        .map(|p| {
            let pc = p.clamp(1e-15, 1.0);
            pc * pc.ln()
        })
        .sum::<f64>()
}

fn build_table<'a, I>(items: I, n_classes: usize) -> PyResult<HashMap<i64, (Vec<f64>, f64)>>
where
    I: Iterator<Item = (i64, i64)> + 'a,
{
    let mut table: HashMap<i64, (Vec<f64>, f64)> = HashMap::new();
    for (key, label) in items {
        if label < 0 || label as usize >= n_classes {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "label {} outside [0, {})",
                label, n_classes
            )));
        }
        let entry = table
            .entry(key)
            .or_insert_with(|| (vec![0.0; n_classes], 0.0));
        entry.0[label as usize] += 1.0;
        entry.1 += 1.0;
    }
    Ok(table)
}

fn fill_posterior_row(
    row: &mut [f64],
    key: i64,
    table: &HashMap<i64, (Vec<f64>, f64)>,
    n_classes: usize,
    alpha: f64,
    global_prior: &[f64],
    prior_entropy: f64,
) {
    let mut entropy = 0.0;
    let n = if let Some((counts, total)) = table.get(&key) {
        for k in 0..n_classes {
            let p = (counts[k] + alpha * global_prior[k]) / (total + alpha);
            row[k] = p;
            let pc = p.clamp(1e-15, 1.0);
            entropy -= pc * pc.ln();
        }
        *total
    } else {
        row[..n_classes].copy_from_slice(&global_prior[..n_classes]);
        entropy = prior_entropy;
        0.0
    };
    row[n_classes] = n.ln_1p();
    row[n_classes + 1] = if n + alpha > 0.0 {
        n / (n + alpha)
    } else {
        0.0
    };
    row[n_classes + 2] = prior_entropy - entropy;
}

fn fill_posterior_from_counts(
    row: &mut [f64],
    counts: &[f64],
    total: f64,
    n_classes: usize,
    alpha: f64,
    global_prior: &[f64],
    prior_entropy: f64,
) {
    if total <= 0.0 {
        row[..n_classes].copy_from_slice(&global_prior[..n_classes]);
        row[n_classes] = 0.0;
        row[n_classes + 1] = 0.0;
        row[n_classes + 2] = 0.0;
        return;
    }

    let mut entropy = 0.0;
    for k in 0..n_classes {
        let p = (counts[k] + alpha * global_prior[k]) / (total + alpha);
        row[k] = p;
        let pc = p.clamp(1e-15, 1.0);
        entropy -= pc * pc.ln();
    }
    row[n_classes] = total.ln_1p();
    row[n_classes + 1] = if total + alpha > 0.0 {
        total / (total + alpha)
    } else {
        0.0
    };
    row[n_classes + 2] = prior_entropy - entropy;
}

fn array2_from_flat<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: Vec<f64>,
    label: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = Array2::from_shape_vec((n_rows, n_cols), data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", label, e))
    })?;
    Ok(arr.into_pyarray(py))
}

fn apply_table_flat(
    keys: &[i64],
    table: &HashMap<i64, (Vec<f64>, f64)>,
    n_classes: usize,
    alpha: f64,
    global_prior: &[f64],
    prior_entropy: f64,
) -> Vec<f64> {
    let n_out_cols = n_classes + 3;
    let mut rows = vec![0.0; keys.len() * n_out_cols];
    for (i, &key) in keys.iter().enumerate() {
        let row = &mut rows[i * n_out_cols..(i + 1) * n_out_cols];
        fill_posterior_row(
            row,
            key,
            table,
            n_classes,
            alpha,
            global_prior,
            prior_entropy,
        );
    }
    rows
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn full_tuple_key(values: &[i64]) -> i64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for (k, &value) in values.iter().enumerate() {
        let pos = (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mixed = splitmix64((value as u64) ^ pos);
        h ^= mixed;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
        h = splitmix64(h);
    }
    (h & 0x7FFF_FFFF_FFFF_FFFF) as i64
}

#[pyfunction]
pub fn pcf_hash_keys<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    cols: Vec<usize>,
    hash_bins: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if cols.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cols must not be empty",
        ));
    }
    if hash_bins < 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hash_bins must be non-negative; use 0 for full tuple keys",
        ));
    }
    let view = x.as_array();
    let shape = view.shape();
    let n_rows = shape[0];
    let n_features = shape[1];
    for &col in &cols {
        if col >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "column {} outside n_features={}",
                col, n_features
            )));
        }
    }
    let mut out = vec![0i64; n_rows];
    if cols.len() == 1 {
        let col = cols[0];
        for i in 0..n_rows {
            out[i] = view[[i, col]] as i64;
        }
    } else if hash_bins == 0 {
        let mut values = vec![0i64; cols.len()];
        for i in 0..n_rows {
            for (k, &col) in cols.iter().enumerate() {
                values[k] = view[[i, col]] as i64;
            }
            out[i] = full_tuple_key(&values);
        }
    } else {
        for i in 0..n_rows {
            let mut h = 0i64;
            for (k, &col) in cols.iter().enumerate() {
                let vals = view[[i, col]] as i64;
                let mult = 1_000_003i64.wrapping_add(2_654_435_761i64.wrapping_mul((k + 1) as i64));
                h ^= vals.wrapping_mul(mult);
            }
            out[i] = h.rem_euclid(hash_bins);
        }
    }
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
pub fn pcf_posterior_apply<'py>(
    py: Python<'py>,
    keys_fit: PyReadonlyArray1<'py, i64>,
    y_fit: PyReadonlyArray1<'py, i64>,
    keys_apply: PyReadonlyArray1<'py, i64>,
    n_classes: usize,
    alpha: f64,
    global_prior: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_prior(n_classes, &global_prior)?;
    let fit_keys = keys_fit.as_slice()?;
    let fit_y = y_fit.as_slice()?;
    if fit_keys.len() != fit_y.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "keys_fit and y_fit must have the same length",
        ));
    }
    let apply_keys = keys_apply.as_slice()?;
    let table = build_table(
        fit_keys.iter().copied().zip(fit_y.iter().copied()),
        n_classes,
    )?;
    let pe = prior_entropy(&global_prior);
    let rows = apply_table_flat(apply_keys, &table, n_classes, alpha, &global_prior, pe);
    array2_from_flat(
        py,
        apply_keys.len(),
        n_classes + 3,
        rows,
        "pcf output array",
    )
}

#[pyfunction]
pub fn pcf_posterior_oof_apply<'py>(
    py: Python<'py>,
    keys_fit: PyReadonlyArray1<'py, i64>,
    y_fit: PyReadonlyArray1<'py, i64>,
    fold_ids: PyReadonlyArray1<'py, i64>,
    keys_apply: PyReadonlyArray1<'py, i64>,
    n_classes: usize,
    alpha: f64,
    global_prior: Vec<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    validate_prior(n_classes, &global_prior)?;
    let fit_keys = keys_fit.as_slice()?;
    let fit_y = y_fit.as_slice()?;
    let folds = fold_ids.as_slice()?;
    let apply_keys = keys_apply.as_slice()?;
    if fit_keys.len() != fit_y.len() || fit_keys.len() != folds.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "keys_fit, y_fit, and fold_ids must have the same length",
        ));
    }
    let pe = prior_entropy(&global_prior);
    let n_out_cols = n_classes + 3;

    let full_table = build_table(
        fit_keys.iter().copied().zip(fit_y.iter().copied()),
        n_classes,
    )?;
    let apply_rows = apply_table_flat(apply_keys, &full_table, n_classes, alpha, &global_prior, pe);

    let mut unique_folds: Vec<i64> = folds.iter().copied().filter(|f| *f >= 0).collect();
    unique_folds.sort_unstable();
    unique_folds.dedup();

    let mut fold_tables: HashMap<i64, HashMap<i64, (Vec<f64>, f64)>> = HashMap::new();
    for fold in &unique_folds {
        fold_tables.insert(*fold, HashMap::new());
    }
    for ((&key, &label), &fold) in fit_keys.iter().zip(fit_y.iter()).zip(folds.iter()) {
        if fold < 0 {
            continue;
        }
        let Some(table) = fold_tables.get_mut(&fold) else {
            continue;
        };
        let entry = table
            .entry(key)
            .or_insert_with(|| (vec![0.0; n_classes], 0.0));
        entry.0[label as usize] += 1.0;
        entry.1 += 1.0;
    }

    let mut oof_rows = vec![0.0; fit_keys.len() * n_out_cols];
    for (i, (&key, &row_fold)) in fit_keys.iter().zip(folds.iter()).enumerate() {
        if row_fold < 0 {
            continue;
        }
        let Some((full_counts, full_total)) = full_table.get(&key) else {
            continue;
        };
        let held_out = fold_tables.get(&row_fold).and_then(|table| table.get(&key));
        let mut counts = full_counts.clone();
        let mut total = *full_total;
        if let Some((held_counts, held_total)) = held_out {
            for k in 0..n_classes {
                counts[k] -= held_counts[k];
            }
            total -= held_total;
        }
        let row = &mut oof_rows[i * n_out_cols..(i + 1) * n_out_cols];
        fill_posterior_from_counts(row, &counts, total, n_classes, alpha, &global_prior, pe);
    }

    let oof = array2_from_flat(
        py,
        fit_keys.len(),
        n_out_cols,
        oof_rows,
        "pcf oof output array",
    )?;
    let apply = array2_from_flat(
        py,
        apply_keys.len(),
        n_out_cols,
        apply_rows,
        "pcf apply output array",
    )?;
    Ok((oof, apply))
}
