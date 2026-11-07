//! The seam between a front end and whatever serves models.
//!
//! LM Studio is the first implementation ([`crate::lmstudio::LmStudio`]); cloud
//! providers are later ones. Front ends hold an `Arc<dyn ModelBackend>`, so tests
//! can swap in a fake that needs no running server.

use std::{future::Future, pin::Pin};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::{
    Message,
    chat::{ChatDelta, RequestOptions},
    lmstudio::ModelInfo,
    tools::ToolSpec,
};

/// A boxed, sendable future. Trait methods return this so the trait stays
/// object-safe (`async fn` in a trait cannot be called through `dyn`).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Something that can be started, asked for its models, told to load or unload
/// one, and chatted with.
pub trait ModelBackend: Send + Sync {
    /// Makes the backend reachable and reports the model already loaded, if any.
    fn start(&self) -> BoxFuture<'_, Result<Option<ModelInfo>>>;

    /// Lists chat-capable models, in the backend's own order.
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>>;

    /// Loads `id` and returns its updated entry once it is ready.
    fn load_model<'a>(&'a self, id: &'a str) -> BoxFuture<'a, Result<ModelInfo>>;

    /// Unloads `id` and resolves once the backend no longer reports it loaded.
    fn unload_model<'a>(&'a self, id: &'a str) -> BoxFuture<'a, Result<()>>;

    /// Sends `messages` to `model`, offering it `tools` (none if empty) under
    /// the limits in `options`, and streams the reply into `sink` as it is
    /// generated. Resolves once the stream ends. To cancel, drop the future
    /// (e.g. abort the task running it).
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: &'a [Message],
        tools: &'a [ToolSpec],
        options: &'a RequestOptions,
        sink: mpsc::Sender<ChatDelta>,
    ) -> BoxFuture<'a, Result<()>>;
}
