//! # RustML Hub
//!
//! HuggingFace Hub integration for loading pre-trained models.
//!
//! This crate provides functionality to:
//! - Download models from HuggingFace Hub
//! - Load SafeTensors format weights
//! - Map weights to RustML model architectures
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustml_hub::HubApi;
//!
//! let api = HubApi::new();
//! let bundle = api.download_model("openai-community/gpt2").await?;
//! let weights = bundle.load_tensors()?;
//! ```

pub mod api;
pub mod safetensors;
pub mod weight_mapper;

pub use api::{HubApi, ModelBundle};
pub use safetensors::{SafeTensorLoader, SafeTensorsError};
pub use weight_mapper::{WeightMapper, Gpt2WeightMapper};

use thiserror::Error;

/// Result type for hub operations
pub type HubResult<T> = Result<T, HubError>;

/// Errors that can occur in hub operations
#[derive(Error, Debug)]
pub enum HubError {
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Weight loading error: {0}")]
    WeightLoadError(String),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(#[from] SafeTensorsError),

    #[error("Tensor error: {0}")]
    TensorError(#[from] rustml_core::TensorError),
}
