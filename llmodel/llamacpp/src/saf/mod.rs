pub use crate::api::loader::LlamaCppBackendLoader;
// `LlamaCppModel`, `LlamaCppTextCompleter`, and `load_llama_cpp_model`
// are deliberately NOT re-exported. The concrete struct is `pub(crate)`
// to enforce the self-referential pool safety invariants (see
// `core/model.rs`); the free loader function is an implementation detail
// of `LlamaCppBackendLoader::load` and has no place on the public surface.
