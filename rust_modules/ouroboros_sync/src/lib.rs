use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Represents a block from Æthel Forge
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct AethelBlock {
    #[pyo3(get, set)]
    pub index: u64,
    #[pyo3(get, set)]
    pub timestamp: f64,
    #[pyo3(get, set)]
    pub data: String,
    #[pyo3(get, set)]
    pub previous_hash: String,
    #[pyo3(get, set)]
    pub hash: String,
    #[pyo3(get, set)]
    pub ergotropy: f64,
}

#[pymethods]
impl AethelBlock {
    #[new]
    pub fn new(
        index: u64,
        timestamp: f64,
        data: String,
        previous_hash: String,
        ergotropy: f64,
    ) -> Self {
        let mut block = AethelBlock {
            index,
            timestamp,
            data,
            previous_hash,
            hash: String::new(),
            ergotropy,
        };
        block.hash = block.calculate_hash();
        block
    }

    /// Calculate the hash of the block
    pub fn calculate_hash(&self) -> String {
        let data = format!(
            "{}{}{}{}{}",
            self.index, self.timestamp, self.data, self.previous_hash, self.ergotropy
        );
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Verify the block's hash
    pub fn verify_hash(&self) -> bool {
        self.hash == self.calculate_hash()
    }
}

/// Configuration for Ouroboros sync
#[derive(Debug, Clone)]
#[pyclass]
pub struct OuroborosSyncConfig {
    #[pyo3(get, set)]
    pub ergotropy_threshold: f64,
    #[pyo3(get, set)]
    pub stability_window: usize,
    #[pyo3(get, set)]
    pub max_recursion_depth: usize,
}

#[pymethods]
impl OuroborosSyncConfig {
    #[new]
    pub fn new(
        ergotropy_threshold: f64,
        stability_window: usize,
        max_recursion_depth: usize,
    ) -> Self {
        OuroborosSyncConfig {
            ergotropy_threshold,
            stability_window,
            max_recursion_depth,
        }
    }
}

impl Default for OuroborosSyncConfig {
    fn default() -> Self {
        OuroborosSyncConfig {
            ergotropy_threshold: 0.5,
            stability_window: 10,
            max_recursion_depth: 5,
        }
    }
}

/// Validation result for a block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ValidationResult {
    #[pyo3(get)]
    pub valid: bool,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub ergotropy_check: bool,
    #[pyo3(get)]
    pub stability_check: bool,
    #[pyo3(get)]
    pub hash_check: bool,
    #[pyo3(get)]
    pub chain_check: bool,
}

#[pymethods]
impl ValidationResult {
    fn __repr__(&self) -> String {
        format!(
            "ValidationResult(valid={}, reason='{}', ergotropy_check={}, stability_check={}, hash_check={}, chain_check={})",
            self.valid, self.reason, self.ergotropy_check, self.stability_check, self.hash_check, self.chain_check
        )
    }
}

/// Main Ouroboros sync consensus mechanism
#[pyclass]
pub struct OuroborosSync {
    config: OuroborosSyncConfig,
    blockchain: Vec<AethelBlock>,
    stability_history: Vec<f64>,
}

#[pymethods]
impl OuroborosSync {
    #[new]
    pub fn new(config: Option<OuroborosSyncConfig>) -> Self {
        OuroborosSync {
            config: config.unwrap_or_default(),
            blockchain: Vec::new(),
            stability_history: Vec::new(),
        }
    }

    /// Validate a block from Æthel Forge
    pub fn validate_block(&mut self, block: &AethelBlock) -> ValidationResult {
        // Check 1: Verify block hash
        let hash_check = block.verify_hash();
        if !hash_check {
            return ValidationResult {
                valid: false,
                reason: "Block hash verification failed".to_string(),
                ergotropy_check: false,
                stability_check: false,
                hash_check,
                chain_check: false,
            };
        }

        // Check 2: Verify chain continuity
        let chain_check = if self.blockchain.is_empty() {
            // Genesis block must have index 0
            if block.index != 0 {
                return ValidationResult {
                    valid: false,
                    reason: format!("Genesis block must have index 0, got {}", block.index),
                    ergotropy_check: false,
                    stability_check: false,
                    hash_check,
                    chain_check: false,
                };
            }
            true
        } else {
            let last_block = self.blockchain.last().unwrap();
            
            // Check index continuity
            if block.index != last_block.index + 1 {
                return ValidationResult {
                    valid: false,
                    reason: format!(
                        "Block index {} does not follow previous index {}",
                        block.index, last_block.index
                    ),
                    ergotropy_check: false,
                    stability_check: false,
                    hash_check,
                    chain_check: false,
                };
            }
            
            // Check hash continuity
            if block.previous_hash != last_block.hash {
                return ValidationResult {
                    valid: false,
                    reason: "Block previous_hash does not match last block hash".to_string(),
                    ergotropy_check: false,
                    stability_check: false,
                    hash_check,
                    chain_check: false,
                };
            }
            
            true
        };

        // Check 3: Ergotropy threshold check
        let ergotropy_check = self.check_ergotropy_threshold(block.ergotropy);
        if !ergotropy_check {
            return ValidationResult {
                valid: false,
                reason: format!(
                    "Ergotropy {} below threshold {}",
                    block.ergotropy, self.config.ergotropy_threshold
                ),
                ergotropy_check,
                stability_check: false,
                hash_check,
                chain_check,
            };
        }

        // Check 4: Recursive stability check
        let stability_check = self.check_recursive_stability(block.ergotropy);
        if !stability_check {
            return ValidationResult {
                valid: false,
                reason: "Recursive stability check failed".to_string(),
                ergotropy_check,
                stability_check,
                hash_check,
                chain_check,
            };
        }

        // All checks passed
        ValidationResult {
            valid: true,
            reason: "Block validated successfully".to_string(),
            ergotropy_check,
            stability_check,
            hash_check,
            chain_check,
        }
    }

    /// Add a validated block to the blockchain
    pub fn add_block(&mut self, block: AethelBlock) -> PyResult<bool> {
        let validation = self.validate_block(&block);
        if validation.valid {
            self.stability_history.push(block.ergotropy);
            // Keep only the last stability_window entries
            if self.stability_history.len() > self.config.stability_window {
                self.stability_history.remove(0);
            }
            self.blockchain.push(block);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the current blockchain length
    pub fn get_chain_length(&self) -> usize {
        self.blockchain.len()
    }

    /// Get the last block in the chain
    pub fn get_last_block(&self) -> Option<AethelBlock> {
        self.blockchain.last().cloned()
    }

    /// Get block by index
    pub fn get_block(&self, index: usize) -> Option<AethelBlock> {
        self.blockchain.get(index).cloned()
    }

    /// Get stability statistics
    pub fn get_stability_stats(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            if self.stability_history.is_empty() {
                dict.set_item("mean", 0.0)?;
                dict.set_item("variance", 0.0)?;
                dict.set_item("min", 0.0)?;
                dict.set_item("max", 0.0)?;
                dict.set_item("count", 0)?;
            } else {
                let mean = self.stability_history.iter().sum::<f64>() / self.stability_history.len() as f64;
                let variance = self.stability_history.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / self.stability_history.len() as f64;
                let min = self.stability_history.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = self.stability_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                
                dict.set_item("mean", mean)?;
                dict.set_item("variance", variance)?;
                dict.set_item("min", min)?;
                dict.set_item("max", max)?;
                dict.set_item("count", self.stability_history.len())?;
            }
            
            Ok(dict.into())
        })
    }

    /// Reset the blockchain
    pub fn reset(&mut self) {
        self.blockchain.clear();
        self.stability_history.clear();
    }
}

impl OuroborosSync {
    /// Check if ergotropy meets the threshold
    fn check_ergotropy_threshold(&self, ergotropy: f64) -> bool {
        ergotropy >= self.config.ergotropy_threshold
    }

    /// Check recursive stability using historical data
    fn check_recursive_stability(&self, current_ergotropy: f64) -> bool {
        if self.stability_history.is_empty() {
            // First block is always considered stable
            return true;
        }

        // Calculate stability recursively with exponential weighting
        self.recursive_stability_check(
            &self.stability_history,
            current_ergotropy,
            0,
            self.config.max_recursion_depth,
        )
    }

    /// Recursive stability check with depth limit
    fn recursive_stability_check(
        &self,
        history: &[f64],
        current: f64,
        depth: usize,
        max_depth: usize,
    ) -> bool {
        if depth >= max_depth || history.is_empty() {
            return true;
        }

        // Calculate simple average of recent history
        let window_size = std::cmp::min(history.len(), self.config.stability_window);
        let recent_history = &history[history.len().saturating_sub(window_size)..];
        
        let avg = recent_history.iter().sum::<f64>() / recent_history.len() as f64;

        // Check if current value is within acceptable deviation
        let deviation = (current - avg).abs();
        // More lenient threshold: 50% deviation or 0.5 absolute, whichever is larger
        let threshold = (0.5 * avg.abs()).max(0.5);

        if deviation > threshold {
            return false;
        }

        // Recursively check with reduced window (less strict at deeper levels)
        if history.len() > 1 && depth < max_depth - 1 {
            let reduced_history = &history[..history.len() - 1];
            self.recursive_stability_check(reduced_history, avg, depth + 1, max_depth)
        } else {
            true
        }
    }
}

/// Python module definition
#[pymodule]
fn ouroboros_sync(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AethelBlock>()?;
    m.add_class::<OuroborosSyncConfig>()?;
    m.add_class::<ValidationResult>()?;
    m.add_class::<OuroborosSync>()?;
    Ok(())
}
