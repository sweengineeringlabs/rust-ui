//! # RustML NLP
//!
//! Natural Language Processing models for RustML, including GPT-2.
//!
//! This crate provides:
//! - GPT-2 model implementation with support for all variants (small, medium, large, xl)
//! - Text generation with temperature, top-k, and top-p sampling
//! - BPE tokenizer for GPT-2
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustml_nlp::{GptModel, GptConfig, TextGenerator, GenerationConfig};
//! use rustml_nlp::tokenizer::BpeTokenizer;
//! use rustml_hub::HubApi;
//!
//! // Load model
//! let api = HubApi::new();
//! let bundle = api.download_model("openai-community/gpt2").await?;
//! let weights = bundle.load_tensors()?;
//! let model = GptModel::from_hub_weights(GptConfig::gpt2_small(), weights)?;
//!
//! // Load tokenizer
//! let tokenizer = BpeTokenizer::from_files("vocab.json", "merges.txt")?;
//!
//! // Generate text
//! let generator = TextGenerator::new(&model);
//! let output = generator.generate(
//!     &tokenizer.encode("Hello world"),
//!     &GenerationConfig::default(),
//! )?;
//!
//! println!("{}", tokenizer.decode(&output));
//! ```

pub mod generation;
pub mod gpt;
pub mod tokenizer;

pub use generation::{GenerationConfig, TextGenerator};
pub use gpt::{GptBlock, GptConfig, GptMlp, GptModel};
pub use tokenizer::BpeTokenizer;

use thiserror::Error;

/// Result type for NLP operations
pub type NlpResult<T> = Result<T, NlpError>;

/// Errors that can occur in NLP operations
#[derive(Error, Debug)]
pub enum NlpError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] rustml_core::TensorError),

    #[error("Neural network error: {0}")]
    NnError(#[from] rustml_nn::NnError),

    #[error("Hub error: {0}")]
    HubError(#[from] rustml_hub::HubError),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Generation error: {0}")]
    GenerationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
