//! Weight loading layer — the `llmweights` boundary.
//!
//! Responsibility: read a model from disk (SafeTensors + config.json +
//! tokenizer.json), map HuggingFace names to GGUF names, classify each
//! tensor, apply the norm +1.0 offset, handle tied embeddings, and
//! produce a `WeightBatch` ready for quantization.
//!
//! No quantization math lives here. The only numeric transformation is
//! the norm weight +1.0 offset, which is a format convention, not a
//! lossy operation.

use swe_llmmodel_download::{Download, HuggingFaceDownload};
use swe_llmmodel_io::{LoadTensors, SafeTensorsStore};

use crate::api::error::{QuantizeError, QuantizeResult};
use crate::api::types::{LoadedWeight, TensorClass, WeightBatch};
use crate::core::classifier::classify_tensor;
use crate::core::metadata::{build_gguf_metadata, build_tokenizer_metadata};
use crate::core::name_map::hf_to_gguf_name;

pub(crate) struct DefaultWeightLoader;

impl DefaultWeightLoader {
    pub(crate) fn new() -> Self { Self }

    /// Load all weights and metadata from a HuggingFace cached model directory.
    pub(crate) fn load(&self, model_id: &str) -> QuantizeResult<WeightBatch> {
        let downloader = HuggingFaceDownload::new();
        let model_dir = downloader.cache_dir().join(model_id.replace('/', "--"));

        let safetensors_path = model_dir.join("model.safetensors");
        let config_path = model_dir.join("config.json");

        if !safetensors_path.exists() {
            return Err(QuantizeError::ModelLoad(format!(
                "model.safetensors not found at {}. Download first with: \
                 rustml hub download {}",
                safetensors_path.display(), model_id
            )));
        }
        if !config_path.exists() {
            return Err(QuantizeError::ModelLoad(format!(
                "config.json not found at {}", config_path.display()
            )));
        }

        let config_json: serde_json::Value = {
            let data = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&data)?
        };

        let model_type = config_json
            .get("model_type")
            .or_else(|| config_json.get("text_config").and_then(|tc| tc.get("model_type")))
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let arch = normalize_arch_str(&model_type);
        log::info!("Model type: {} (arch: {})", model_type, arch);

        // Build GGUF metadata
        let mut metadata = build_gguf_metadata(&config_json)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            log::info!("Loading tokenizer from {}", tokenizer_path.display());
            let tok_data = std::fs::read_to_string(&tokenizer_path)?;
            let tok_json: serde_json::Value = serde_json::from_str(&tok_data)?;
            metadata.extend(build_tokenizer_metadata(&tok_json)?);
        } else {
            log::warn!("No tokenizer.json found — GGUF will lack tokenizer metadata");
        }

        // Tied embeddings flag
        let tie_embeddings = config_json
            .get("tie_word_embeddings")
            .or_else(|| config_json.get("text_config").and_then(|tc| tc.get("tie_word_embeddings")))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Load SafeTensors weights
        log::info!("Loading safetensors from {}", safetensors_path.display());
        let tensors = SafeTensorsStore.load(&safetensors_path)?;
        log::info!("Loaded {} tensors", tensors.len());

        // Sort for deterministic output
        let mut tensor_names: Vec<&String> = tensors.keys().collect();
        tensor_names.sort();

        let mut weights = Vec::with_capacity(tensor_names.len());
        let mut has_output_weight = false;
        let mut emb_weight_for_tie: Option<(Vec<f32>, Vec<usize>)> = None;

        for hf_name in tensor_names {
            if is_multimodal_tensor(hf_name) {
                log::debug!("Skipping multimodal tensor: {}", hf_name);
                continue;
            }

            let tensor = &tensors[hf_name];
            let shape = tensor.shape().to_vec();
            let original_bytes = tensor.as_raw_bytes().map(|b| b.len()).unwrap_or(0) as u64;
            let class = classify_tensor(hf_name);
            let gguf_name = hf_to_gguf_name(hf_name, &model_type);

            if gguf_name == "output.weight" { has_output_weight = true; }

            // Apply norm +1.0 offset (GGUF convention for RMSNorm weights)
            let data_f32 = if class == TensorClass::Norm || gguf_name.contains("norm") {
                let mut data = tensor.to_f32()?.to_vec();
                for v in &mut data { *v += 1.0; }
                data
            } else {
                tensor.to_f32()?.to_vec()
            };

            // Track embedding for tied embedding duplication
            if gguf_name == "token_embd.weight" && tie_embeddings {
                emb_weight_for_tie = Some((data_f32.clone(), shape.clone()));
            }

            weights.push(LoadedWeight {
                hf_name: hf_name.clone(),
                gguf_name,
                class,
                data_f32,
                shape,
                original_bytes,
            });
        }

        // Duplicate token_embd as output.weight for tied embedding models
        if tie_embeddings && !has_output_weight {
            if let Some((data_f32, shape)) = emb_weight_for_tie {
                let original_bytes = (data_f32.len() * 4) as u64;
                log::info!("Tied embeddings: duplicating token_embd.weight as output.weight");
                weights.push(LoadedWeight {
                    hf_name: "token_embd.weight (tied)".to_string(),
                    gguf_name: "output.weight".to_string(),
                    class: TensorClass::Output,
                    data_f32,
                    shape,
                    original_bytes,
                });
            }
        }

        Ok(WeightBatch { arch, metadata, weights, tie_embeddings })
    }
}

fn normalize_arch_str(model_type: &str) -> String {
    match model_type {
        "gemma4" | "gemma4_text" | "gemma3" | "gemma3_text" | "gemma2" => "gemma2".to_string(),
        "gemma" => "gemma".to_string(),
        "llama" | "mistral" => "llama".to_string(),
        "qwen2" => "qwen2".to_string(),
        "phi3" | "phi" => "phi3".to_string(),
        other => other.to_string(),
    }
}

fn is_multimodal_tensor(name: &str) -> bool {
    name.contains("vision_tower")
        || name.contains("audio_tower")
        || name.contains("embed_vision")
        || name.contains("embed_audio")
        || name.contains("multi_modal_projector")
        || name.contains("image_encoder")
        || name.contains("audio_encoder")
}
