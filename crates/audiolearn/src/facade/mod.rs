//! Facade Layer - App entry points and UI components

mod app;
mod components;
mod pages;

#[cfg(test)]
mod create_tests;

pub use app::*;
pub use components::*;
pub use pages::*;
