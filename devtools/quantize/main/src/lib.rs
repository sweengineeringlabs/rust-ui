// Thin re-export — all logic lives in llmweights and llmquant.
pub use llmquant::{DefaultQuantizeEngine, QuantizeConfig, QuantizeEngine, QuantizeReport, QuantTarget};
pub use llmweights::WeightBatch;

pub fn create_engine() -> DefaultQuantizeEngine {
    DefaultQuantizeEngine::new()
}
