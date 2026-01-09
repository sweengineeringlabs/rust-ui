//! # RustML Neural Network
//!
//! Neural network layers and modules for RustML.
//!
//! This crate provides building blocks for neural networks including:
//! - Linear layers
//! - Embedding layers
//! - Layer normalization
//! - Attention mechanisms (including causal self-attention for GPT)
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustml_nn::{Linear, LayerNorm, CausalSelfAttention};
//! use rustml_core::Tensor;
//!
//! let linear = Linear::new(768, 768);
//! let output = linear.forward(&input)?;
//! ```

pub mod attention;
pub mod embedding;
pub mod layer_norm;
pub mod linear;

pub use attention::{Attention, CausalSelfAttention, MultiHeadAttention};
pub use embedding::Embedding;
pub use layer_norm::LayerNorm;
pub use linear::Linear;

use rustml_core::TensorError;
use thiserror::Error;

/// Result type for neural network operations
pub type NnResult<T> = Result<T, NnError>;

/// Errors that can occur in neural network operations
#[derive(Error, Debug)]
pub enum NnError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Weight initialization error: {0}")]
    WeightInitError(String),
}
