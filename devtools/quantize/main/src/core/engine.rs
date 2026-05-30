//! Quantization engine — orchestrates loader → quantizer → writer.
//!
//! This module is intentionally thin. All weight loading logic lives in
//! `loader.rs` (`llmweights` boundary); all quantization math lives in
//! `quantizer.rs` (`llmquant` boundary). The engine wires them together
//! and writes the GGUF output.

use swe_llmmodel_gguf::GGMLType;

use crate::api::error::QuantizeResult;
use crate::api::traits::QuantizeEngine;
use crate::api::types::{QuantizeConfig, QuantizeReport, TensorClass, TensorReport};
use crate::core::loader::DefaultWeightLoader;
use crate::core::metrics::compute_metrics;
use crate::core::quantizer::{dequantize, quantize_f32, should_quantize};

pub(crate) struct DefaultQuantizeEngine;

impl DefaultQuantizeEngine {
    pub(crate) fn new() -> Self { Self }
}

impl QuantizeEngine for DefaultQuantizeEngine {
    fn run(&self, config: &QuantizeConfig) -> QuantizeResult<QuantizeReport> {
        // ── Load ──────────────────────────────────────────────────────────
        let loader = DefaultWeightLoader::new();
        let mut batch = loader.load(&config.model_id)?;

        // Update file_type metadata based on target
        let file_type_value = match config.target {
            crate::api::types::QuantTarget::Q8_0 => 7u32,
            crate::api::types::QuantTarget::Q4_0 => 2u32,
            crate::api::types::QuantTarget::Q4_1 => 3u32,
        };
        for (key, val) in &mut batch.metadata {
            if key == "general.file_type" {
                *val = swe_llmmodel_gguf::GGUFValue::U32(file_type_value);
            }
        }

        // ── Quantize + Write ──────────────────────────────────────────────
        let mut writer = swe_llmmodel_gguf::writer::GGUFWriter::new();
        for (key, value) in &batch.metadata {
            writer.add_metadata(key.clone(), value.clone());
        }

        let mut report = QuantizeReport {
            total_tensors: batch.weights.len(),
            quantized_tensors: 0,
            skipped_tensors: 0,
            original_bytes: 0,
            quantized_bytes: 0,
            compression_ratio: 1.0,
            per_tensor: Vec::new(),
        };

        for w in &batch.weights {
            report.original_bytes += w.original_bytes;
            let dims = gguf_dims(&w.shape);
            let is_output = w.class == TensorClass::Output;

            let do_quantize = should_quantize(
                w.class.should_quantize(),
                &w.shape,
                config.min_dim,
                config.preserve_output,
                is_output,
            );

            if do_quantize {
                let (quantized_data, ggml_type) = quantize_f32(&w.data_f32, config.target)?;
                let quantized_bytes = quantized_data.len() as u64;
                report.quantized_bytes += quantized_bytes;
                report.quantized_tensors += 1;

                let tensor_report = if config.show_metrics {
                    let dequantized = dequantize(&quantized_data, w.data_f32.len(), config.target)?;
                    compute_metrics(
                        &w.hf_name, &w.data_f32, &dequantized,
                        "F32", config.target.label(),
                        w.original_bytes, quantized_bytes,
                    )
                } else {
                    TensorReport {
                        name: w.hf_name.clone(),
                        original_dtype: "F32".into(),
                        target_dtype: config.target.label().to_string(),
                        original_bytes: w.original_bytes,
                        quantized_bytes,
                        mse: None, max_abs_error: None, snr_db: None,
                    }
                };

                if config.show_metrics {
                    log::info!(
                        "  {}: F32 -> {} ({:.1}x) MSE={:.2e} SNR={:.1}dB",
                        w.hf_name,
                        config.target.label(),
                        w.original_bytes as f64 / quantized_bytes as f64,
                        tensor_report.mse.unwrap_or(0.0),
                        tensor_report.snr_db.unwrap_or(0.0),
                    );
                } else {
                    log::info!(
                        "  {}: F32 -> {} ({:.1}x)",
                        w.hf_name, config.target.label(),
                        w.original_bytes as f64 / quantized_bytes as f64,
                    );
                }

                report.per_tensor.push(tensor_report);
                writer.add_tensor(w.gguf_name.clone(), dims, ggml_type, quantized_data);
            } else {
                // Keep as F16 (already converted in loader; write as F16 bytes)
                let f16_bytes: Vec<u8> = w.data_f32.iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                    .collect();
                let qbytes = f16_bytes.len() as u64;
                report.quantized_bytes += qbytes;
                report.skipped_tensors += 1;
                log::info!("  {}: kept F16 ({:?}) [{}]", w.hf_name, w.class, w.gguf_name);
                writer.add_tensor(w.gguf_name.clone(), dims, GGMLType::F16, f16_bytes);
            }
        }

        // ── Write GGUF ────────────────────────────────────────────────────
        log::info!("Writing GGUF to {}", config.output_path.display());
        let bytes_written = writer.write_to_file(&config.output_path)?;
        log::info!("Wrote {} GGUF file", format_bytes(bytes_written));

        report.compression_ratio = if report.quantized_bytes > 0 {
            report.original_bytes as f64 / report.quantized_bytes as f64
        } else {
            1.0
        };

        log::info!("=== Quantization Summary ===");
        log::info!("  Total tensors:    {}", report.total_tensors);
        log::info!("  Quantized:        {}", report.quantized_tensors);
        log::info!("  Skipped (F16):    {}", report.skipped_tensors);
        log::info!("  Original size:    {}", format_bytes(report.original_bytes));
        log::info!("  Quantized size:   {}", format_bytes(report.quantized_bytes));
        log::info!("  Compression:      {:.2}x", report.compression_ratio);

        Ok(report)
    }
}

fn gguf_dims(shape: &[usize]) -> Vec<usize> {
    shape.iter().rev().copied().collect()
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
