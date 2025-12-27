//! Core Layer - Business logic implementations
//! 
//! Contains implementations of the API services.

mod course_service;
mod sample_data;
mod rodio_player;
mod audio_test;
mod audio_context;

// TTS implementations
mod native_tts;
mod edge_tts;
mod tts_manager;

#[cfg(test)]
mod tts_tests;

pub use course_service::*;
pub use sample_data::*;
pub use rodio_player::*;
pub use audio_test::*;
pub use audio_context::*;

// TTS exports
pub use native_tts::*;
pub use edge_tts::*;
pub use tts_manager::*;

