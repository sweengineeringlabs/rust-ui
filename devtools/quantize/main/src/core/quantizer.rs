//! Quantization math layer — the `llmquant` boundary.
//!
//! Responsibility: given f32 data, produce quantized bytes in the target
//! format. Also dequantizes for metric computation. No I/O, no GGUF
//! writing, no name mapping. Pure numeric transformation.

use swe_llmmodel_gguf::GGMLType;

use crate::api::error::{QuantizeError, QuantizeResult};
use crate::api::types::QuantTarget;

/// Quantize a flat f32 slice to the target format.
/// Returns the quantized bytes and the corresponding GGML type tag.
pub(crate) fn quantize_f32(data: &[f32], target: QuantTarget) -> QuantizeResult<(Vec<u8>, GGMLType)> {
    match target {
        QuantTarget::Q8_0 => {
            let q = swe_llmmodel_kernel::quantize_q8_0(data)?;
            Ok((q, GGMLType::Q8_0))
        }
        QuantTarget::Q4_0 => {
            let q = swe_llmmodel_kernel::quantize_q4_0(data)?;
            Ok((q, GGMLType::Q4_0))
        }
        QuantTarget::Q4_1 => {
            let q = swe_llmmodel_kernel::quantize_q4_1(data)?;
            Ok((q, GGMLType::Q4_1))
        }
    }
}

/// Dequantize bytes back to f32 for metric computation.
pub(crate) fn dequantize(data: &[u8], n_elements: usize, target: QuantTarget) -> QuantizeResult<Vec<f32>> {
    match target {
        QuantTarget::Q8_0 => Ok(swe_llmmodel_kernel::dequantize_q8_0(data, n_elements)?),
        QuantTarget::Q4_0 => Ok(swe_llmmodel_kernel::dequantize_q4_0(data, n_elements)?),
        QuantTarget::Q4_1 => Ok(swe_llmmodel_kernel::dequantize_q4_1(data, n_elements)?),
    }
}

/// Whether a tensor should be quantized given its shape and config.
pub(crate) fn should_quantize(
    class_quantizable: bool,
    shape: &[usize],
    min_dim: usize,
    preserve_output: bool,
    is_output: bool,
) -> bool {
    class_quantizable
        && shape.len() >= 2
        && shape.iter().copied().max().unwrap_or(0) >= min_dim
        && !(preserve_output && is_output)
        && shape.iter().product::<usize>() % 32 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_q8_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let (q, _) = quantize_f32(&data, QuantTarget::Q8_0).unwrap();
        let back = dequantize(&q, 64, QuantTarget::Q8_0).unwrap();
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.05, "Q8_0 roundtrip: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_should_quantize_requires_2d() {
        assert!(!should_quantize(true, &[1024], 0, false, false));
        assert!(should_quantize(true, &[1024, 1024], 0, false, false));
    }

    #[test]
    fn test_should_quantize_respects_preserve_output() {
        assert!(!should_quantize(true, &[1024, 1024], 0, true, true));
        assert!(should_quantize(true, &[1024, 1024], 0, true, false));
    }

    #[test]
    fn test_should_quantize_respects_min_dim() {
        assert!(!should_quantize(true, &[128, 256], 512, false, false));
        assert!(should_quantize(true, &[512, 256], 512, false, false));
    }

    #[test]
    fn test_should_quantize_requires_class_quantizable() {
        assert!(!should_quantize(false, &[1024, 1024], 0, false, false));
    }
}
