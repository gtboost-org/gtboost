use pyo3::prelude::*;
use pyo3::types::PyModule;

mod helpers;
mod model;
mod pcf;
mod tree;

#[pymodule]
fn gtboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::GTBoostModel>()?;
    m.add_class::<model::GTBoostDataset>()?;
    m.add_function(wrap_pyfunction!(pcf::pcf_hash_keys, m)?)?;
    m.add_function(wrap_pyfunction!(pcf::pcf_posterior_apply, m)?)?;
    m.add_function(wrap_pyfunction!(pcf::pcf_posterior_oof_apply, m)?)?;
    Ok(())
}
