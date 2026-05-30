use std::path::PathBuf;
use swe_llmmodel_gguf::GGUFValue;
use swe_ml_tensor::Tensor;

/// Target quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTarget {
    Q8_0,
    Q4_0,
    Q4_1,
}

impl QuantTarget {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "q8_0" | "q8" => Some(QuantTarget::Q8_0),
            "q4_0" | "q4" => Some(QuantTarget::Q4_0),
            "q4_1" => Some(QuantTarget::Q4_1),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            QuantTarget::Q8_0 => "Q8_0",
            QuantTarget::Q4_0 => "Q4_0",
            QuantTarget::Q4_1 => "Q4_1",
        }
    }
}

/// Classification of a tensor by its role in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorClass {
    Embedding,
    Attention,
    FeedForward,
    Norm,
    Output,
    Gate,
    Unknown,
}

impl TensorClass {
    /// Whether this tensor class should be quantized.
    pub fn should_quantize(&self) -> bool {
        matches!(self, TensorClass::Attention | TensorClass::FeedForward | TensorClass::Output | TensorClass::Gate)
    }
}

/// Configuration for a quantization run.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// HuggingFace model ID or local path to safetensors file.
    pub model_id: String,
    /// Target quantization format.
    pub target: QuantTarget,
    /// Output GGUF file path.
    pub output_path: PathBuf,
    /// Minimum dimension for a tensor to be quantized.
    pub min_dim: usize,
    /// Keep the output/lm_head layer in F32.
    pub preserve_output: bool,
    /// Print per-tensor quality metrics.
    pub show_metrics: bool,
}

/// Per-tensor quantization report.
#[derive(Debug, Clone)]
pub struct TensorReport {
    pub name: String,
    pub original_dtype: String,
    pub target_dtype: String,
    pub original_bytes: u64,
    pub quantized_bytes: u64,
    pub mse: Option<f64>,
    pub max_abs_error: Option<f64>,
    pub snr_db: Option<f64>,
}

/// A single weight loaded from disk, ready for quantization.
/// This is the handoff type between the loader (`llmweights`) and the
/// quantizer (`llmquant`) layers.
pub struct LoadedWeight {
    /// Original HuggingFace tensor name.
    pub hf_name: String,
    /// Mapped GGUF tensor name.
    pub gguf_name: String,
    /// Role of this tensor in the model.
    pub class: TensorClass,
    /// Weight data as f32. Always f32 at this boundary — dtype conversion
    /// happens in the loader; quantization math happens in the quantizer.
    pub data_f32: Vec<f32>,
    /// Original shape (outermost-first).
    pub shape: Vec<usize>,
    /// Original byte size before quantization.
    pub original_bytes: u64,
}

/// All weights loaded from a model directory, plus GGUF metadata.
/// Output of the loader, input to the quantizer.
pub struct WeightBatch {
    /// Architecture string (e.g. "gemma2", "llama").
    pub arch: String,
    /// GGUF key-value metadata from config.json + tokenizer.json.
    pub metadata: Vec<(String, GGUFValue)>,
    /// Weights in deterministic order (sorted by HF name).
    pub weights: Vec<LoadedWeight>,
    /// Whether the model uses tied embeddings (token_embd = output).
    pub tie_embeddings: bool,
}

/// Summary report for a quantization run.
#[derive(Debug, Clone)]
pub struct QuantizeReport {
    pub total_tensors: usize,
    pub quantized_tensors: usize,
    pub skipped_tensors: usize,
    pub original_bytes: u64,
    pub quantized_bytes: u64,
    pub compression_ratio: f64,
    pub per_tensor: Vec<TensorReport>,
}
