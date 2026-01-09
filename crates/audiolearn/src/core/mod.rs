//! Core Layer - Business logic implementations
//! 
//! Contains implementations of the API services.

mod course_service;
mod sample_data;
mod playback_state;
mod search;
mod settings;

#[cfg(feature = "desktop")]
mod rodio_player;
#[cfg(feature = "desktop")]
mod audio_test;
#[cfg(feature = "desktop")]
mod audio_context;

// Desktop TTS implementations
#[cfg(feature = "desktop")]
mod native_tts;
#[cfg(feature = "desktop")]
mod edge_tts;
#[cfg(feature = "desktop")]
mod tts_manager;

// Web TTS implementation
#[cfg(feature = "web")]
mod web_tts;

#[cfg(test)]
mod tts_tests;

pub use course_service::*;
pub use sample_data::*;
pub use playback_state::*;
pub use search::*;
pub use settings::*;

#[cfg(feature = "desktop")]
pub use rodio_player::*;
#[cfg(feature = "desktop")]
pub use audio_test::*;
#[cfg(feature = "desktop")]
pub use audio_context::*;

// TTS exports - platform specific
#[cfg(feature = "desktop")]
pub use native_tts::*;
#[cfg(feature = "desktop")]
pub use edge_tts::*;
#[cfg(feature = "desktop")]
pub use tts_manager::*;

#[cfg(feature = "web")]
pub use web_tts::*;

// Platform-agnostic TTS functions
/// Speak text using the appropriate TTS for the platform
pub fn speak_text(text: &str) -> crate::common::Result<()> {
    #[cfg(feature = "desktop")]
    {
        // Use native TTS on desktop
        let mut tts = native_tts::NativeTts::default();
        use crate::spi::tts::TtsEngine;
        tts.speak(text, &crate::spi::tts::SpeechOptions::default())
    }
    #[cfg(feature = "web")]
    {
        web_tts::web_speak_text(text)
    }
    #[cfg(not(any(feature = "desktop", feature = "web")))]
    {
        let _ = text;
        Err(crate::common::AudioLearnError::Tts("No TTS available".into()))
    }
}

/// Stop TTS playback
pub fn stop_tts() -> crate::common::Result<()> {
    #[cfg(feature = "desktop")]
    {
        let mut tts = native_tts::NativeTts::default();
        use crate::spi::tts::TtsEngine;
        tts.stop()
    }
    #[cfg(feature = "web")]
    {
        web_tts::web_stop_tts()
    }
    #[cfg(not(any(feature = "desktop", feature = "web")))]
    {
        Ok(())
    }
}

/// Get available TTS voices
pub fn get_tts_voices() -> crate::common::Result<Vec<crate::spi::tts::Voice>> {
    #[cfg(feature = "desktop")]
    {
        tts_manager::get_tts_voices()
    }
    #[cfg(feature = "web")]
    {
        let tts = web_tts::WebTts::default();
        use crate::spi::tts::TtsEngine;
        tts.voices()
    }
    #[cfg(not(any(feature = "desktop", feature = "web")))]
    {
        Ok(Vec::new())
    }
}
