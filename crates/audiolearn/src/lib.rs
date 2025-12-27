//! AudioLearn - Audio-first Online Course Platform
//! 
//! Architecture: SEA (Stratified Encapsulation Architecture)
//! 
//! Layers:
//! - common: Shared types, utilities, errors
//! - spi: Service Provider Interfaces (traits for external services)
//! - api: Public API contracts (traits for internal use)
//! - core: Business logic implementations
//! - facade: Main app entry points and composition

pub mod common;
pub mod spi;
pub mod api;
pub mod core;
pub mod facade;

pub mod prelude {
    pub use crate::common::*;
    pub use crate::api::*;
    pub use crate::facade::*;
}
