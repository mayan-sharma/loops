//! Control of a local LM Studio install: start its server, list models, load
//! one, and chat with it.
//!
//! Two channels are used. The `lms` CLI performs actions (start, load, unload)
//! because it blocks until they finish and reports failure by exit code. The
//! server's REST API (`GET /api/v0/models`) reports state, because it is
//! structured and cheap to poll. Chat goes through the OpenAI-compatible
//! `POST /v1/chat/completions` (see [`crate::chat`]).

use std::{
    process::Stdio,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use tokio::{process::Command, sync::mpsc};

use crate::{
    Message,
    backend::{BoxFuture, ModelBackend},
    chat::{ChatDelta, SseParser, parse_chunk, request_body},
    tools::ToolSpec,
};

/// Where `lms server start` listens unless configured otherwise.
pub const DEFAULT_BASE_URL: &str = "http://localhost:1234";

/// Delay between readiness checks while the server starts.
const POLL_INTERVAL: Duration = Duration::from_millis(250);
/// How long to wait for the server to start and then answer.
const START_TIMEOUT: Duration = Duration::from_secs(15);
/// Cap on a single REST call, so a wedged server cannot hang the caller.
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
/// How much `lms` output to keep in an error message.
const ERROR_TAIL_LINES: usize = 20;

/// Whether a model is in memory and ready to answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub enum ModelState {
    /// In memory and ready.
    #[serde(rename = "loaded")]
    Loaded,
    /// On disk only. Any state string other than `loaded` maps here, since only
    /// `loaded` means the model can be used.
    #[default]
    #[serde(rename = "not-loaded", other)]
    NotLoaded,
}

/// One entry of `GET /api/v0/models`. Fields the harness does not use are ignored.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelInfo {
    /// Model key, e.g. `qwen/qwen3.5-9b`. Passed to `lms load`.
    pub id: String,
    /// `llm`, `vlm` or `embeddings` (the API's `type` field).
    #[serde(rename = "type", default)]
    pub kind: String,
    /// Architecture family, e.g. `qwen35`.
    pub arch: Option<String>,
    /// Quantization, e.g. `Q6_K`.
    pub quantization: Option<String>,
    /// Loaded or not.
    #[serde(default)]
    pub state: ModelState,
    /// Largest context window the model supports, in tokens.
    pub max_context_length: Option<u64>,
    /// Context window the model was loaded with, in tokens; usually smaller
    /// than `max_context_length`. Only loaded models report it.
    #[serde(default)]
    pub loaded_context_length: Option<u64>,
    /// Extra abilities such as `tool_use`.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl ModelInfo {
    /// True for models that can chat (everything except embedding models).
    pub fn is_chat(&self) -> bool {
        self.kind != "embeddings"
    }

    /// The context window the model actually has: the loaded length when it
    /// is loaded, else the largest it supports.
    pub fn context_length(&self) -> Option<u64> {
        self.loaded_context_length.or(self.max_context_length)
    }
}

/// Body of `GET /api/v0/models`.
#[derive(Debug, Deserialize)]
struct ModelList {
    data: Vec<ModelInfo>,
}

impl ModelList {
    /// Chat-capable models, in API order.
    fn into_chat_models(self) -> Vec<ModelInfo> {
        self.data.into_iter().filter(ModelInfo::is_chat).collect()
    }
}

/// Client for one LM Studio server plus the `lms` CLI on PATH.
#[derive(Debug, Clone)]
pub struct LmStudio {
    base_url: String,
    http: reqwest::Client,
}

impl Default for LmStudio {
    fn default() -> Self {
        Self::new(DEFAULT_BASE_URL)
    }
}

impl LmStudio {
    /// Creates a client for the server at `base_url` (e.g. `http://localhost:1234`).
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_owned();
        Self {
            base_url,
            http: reqwest::Client::new(),
        }
    }

    /// Makes sure the server answers, running `lms server start` if it does not.
    pub async fn ensure_server(&self) -> Result<()> {
        if self.fetch_models().await.is_ok() {
            return Ok(());
        }

        // Output is discarded rather than piped: `lms server start` may leave a
        // background process holding inherited pipe handles, and reading them
        // to EOF would then never finish.
        let start = Command::new("lms")
            .args(["server", "start"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status();
        let status = tokio::time::timeout(START_TIMEOUT, start)
            .await
            .map_err(|_| anyhow!("`lms server start` did not finish within {START_TIMEOUT:?}"))?
            .map_err(|err| spawn_error(err, "server start"))?;
        if !status.success() {
            bail!("`lms server start` failed ({status})");
        }

        let deadline = Instant::now() + START_TIMEOUT;
        loop {
            match self.fetch_models().await {
                Ok(_) => return Ok(()),
                Err(err) if Instant::now() >= deadline => {
                    return Err(err.context(format!(
                        "LM Studio server at {} did not answer within {START_TIMEOUT:?}",
                        self.base_url
                    )));
                }
                Err(_) => tokio::time::sleep(POLL_INTERVAL).await,
            }
        }
    }

    /// Chat-capable models (embedding models removed), in the order the API returns.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(self.fetch_models().await?.into_chat_models())
    }

    /// The first chat model reported as loaded, if any.
    pub async fn loaded_model(&self) -> Result<Option<ModelInfo>> {
        Ok(self
            .list_models()
            .await?
            .into_iter()
            .find(|m| m.state == ModelState::Loaded))
    }

    /// Loads `id` with `lms load` and returns its entry once the server reports it loaded.
    pub async fn load_model(&self, id: &str) -> Result<ModelInfo> {
        run_lms(&["load", id, "-y"]).await?;
        let model = self
            .list_models()
            .await?
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| {
                anyhow!("`lms load {id}` succeeded but the server does not list {id}")
            })?;
        if model.state != ModelState::Loaded {
            bail!("`lms load {id}` succeeded but the server reports {id} as not loaded");
        }
        Ok(model)
    }

    /// Unloads `id` with `lms unload` and checks the server no longer reports it loaded.
    pub async fn unload_model(&self, id: &str) -> Result<()> {
        run_lms(&["unload", id]).await?;
        let still_loaded = self
            .list_models()
            .await?
            .iter()
            .any(|m| m.id == id && m.state == ModelState::Loaded);
        if still_loaded {
            bail!("`lms unload {id}` succeeded but the server still reports {id} as loaded");
        }
        Ok(())
    }

    /// Unloads every model with `lms unload --all`.
    pub async fn unload_all(&self) -> Result<()> {
        run_lms(&["unload", "--all"]).await.map(drop)
    }

    /// Streams a reply from `model` into `sink`. One `Done` is sent; a `Usage`
    /// may follow it, since the server reports usage after the reply ends.
    /// `tools` are offered to the model; pass `&[]` for plain chat.
    ///
    /// No overall timeout: a long reply is legitimately slow, and the caller
    /// cancels by dropping this future. Returns early if `sink` is closed.
    pub async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        sink: mpsc::Sender<ChatDelta>,
    ) -> Result<()> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = request_body(model, messages, tools);
        let mut response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!("POST {url} returned {status}: {}", text.trim());
        }

        let mut parser = SseParser::new();
        let mut done = false;
        while let Some(bytes) = response
            .chunk()
            .await
            .with_context(|| format!("reading the reply from {url}"))?
        {
            for payload in parser.push(&bytes) {
                let last = payload.trim() == "[DONE]";
                for delta in parse_chunk(&payload)? {
                    // The `finish_reason` chunk and `[DONE]` both say done;
                    // only the first is passed on. Reading goes on after it,
                    // because the usage chunk comes between the two.
                    if matches!(delta, ChatDelta::Done { .. }) {
                        if done {
                            continue;
                        }
                        done = true;
                    }
                    if sink.send(delta).await.is_err() {
                        // Nobody is listening any more.
                        return Ok(());
                    }
                }
                if last {
                    return Ok(());
                }
            }
        }
        if !done {
            bail!("the reply from {url} ended before it said it was done");
        }
        Ok(())
    }

    /// One call to the models endpoint, all entries included.
    async fn fetch_models(&self) -> Result<ModelList> {
        let url = format!("{}/api/v0/models", self.base_url);
        self.http
            .get(&url)
            .timeout(HTTP_TIMEOUT)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .with_context(|| format!("GET {url}"))?
            .json::<ModelList>()
            .await
            .with_context(|| format!("unexpected response from {url}"))
    }
}

impl ModelBackend for LmStudio {
    fn start(&self) -> BoxFuture<'_, Result<Option<ModelInfo>>> {
        Box::pin(async move {
            self.ensure_server().await?;
            self.loaded_model().await
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>> {
        Box::pin(LmStudio::list_models(self))
    }

    fn load_model<'a>(&'a self, id: &'a str) -> BoxFuture<'a, Result<ModelInfo>> {
        Box::pin(LmStudio::load_model(self, id))
    }

    fn unload_model<'a>(&'a self, id: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(LmStudio::unload_model(self, id))
    }

    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: &'a [Message],
        tools: &'a [ToolSpec],
        sink: mpsc::Sender<ChatDelta>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(LmStudio::chat(self, model, messages, tools, sink))
    }
}

/// Runs `lms <args>` to completion. On failure the error carries the tail of its output.
async fn run_lms(args: &[&str]) -> Result<String> {
    let command = args.join(" ");
    let output = Command::new("lms")
        .args(args)
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|err| spawn_error(err, &command))?;

    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        bail!(
            "`lms {command}` failed ({}):\n{}",
            output.status,
            tail(&combined, ERROR_TAIL_LINES)
        );
    }
    Ok(combined)
}

/// Turns a failure to launch `lms` into a message that says what to do.
fn spawn_error(err: std::io::Error, command: &str) -> anyhow::Error {
    if err.kind() == std::io::ErrorKind::NotFound {
        anyhow!("`lms` not found on PATH; install LM Studio and run its `lms bootstrap`")
    } else {
        anyhow!(err).context(format!("could not run `lms {command}`"))
    }
}

/// The last `n` non-empty lines of `text`. Progress bars redraw with `\r`, so
/// that counts as a line break too.
fn tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text
        .split(['\r', '\n'])
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like the live endpoint: an embeddings entry, a loaded model, an
    /// entry missing optional fields, and fields the harness does not model.
    const FIXTURE: &str = r#"{
      "data": [
        { "id": "qwen/qwen3.5-9b", "object": "model", "type": "vlm", "publisher": "qwen",
          "arch": "qwen35", "compatibility_type": "gguf", "quantization": "Q6_K",
          "state": "not-loaded", "max_context_length": 262144, "capabilities": ["tool_use"] },
        { "id": "text-embedding-nomic-embed-text-v1.5", "object": "model", "type": "embeddings",
          "publisher": "nomic-ai", "arch": "nomic-bert", "quantization": "Q4_K_M",
          "state": "not-loaded", "max_context_length": 2048 },
        { "id": "google/gemma-4-e4b", "object": "model", "type": "vlm", "arch": "gemma4",
          "quantization": "Q4_K_M", "state": "loaded", "max_context_length": 131072,
          "loaded_context_length": 8192, "capabilities": ["tool_use"] },
        { "id": "mystery/model", "type": "llm", "state": "loading" }
      ],
      "object": "list"
    }"#;

    fn fixture_models() -> Vec<ModelInfo> {
        serde_json::from_str::<ModelList>(FIXTURE)
            .unwrap()
            .into_chat_models()
    }

    #[test]
    fn embeddings_are_filtered_and_order_is_kept() {
        let ids: Vec<_> = fixture_models().into_iter().map(|m| m.id).collect();
        assert_eq!(
            ids,
            ["qwen/qwen3.5-9b", "google/gemma-4-e4b", "mystery/model"]
        );
    }

    #[test]
    fn fields_map_from_the_api_names() {
        let models = fixture_models();
        let qwen = &models[0];
        assert_eq!(qwen.kind, "vlm");
        assert_eq!(qwen.arch.as_deref(), Some("qwen35"));
        assert_eq!(qwen.quantization.as_deref(), Some("Q6_K"));
        assert_eq!(qwen.state, ModelState::NotLoaded);
        assert_eq!(qwen.max_context_length, Some(262_144));
        assert_eq!(qwen.capabilities, ["tool_use"]);
        assert_eq!(models[1].state, ModelState::Loaded);
    }

    #[test]
    fn context_length_is_the_loaded_one_else_the_max() {
        let models = fixture_models();
        let (qwen, gemma) = (&models[0], &models[1]);
        assert_eq!(gemma.loaded_context_length, Some(8192));
        assert_eq!(gemma.context_length(), Some(8192));
        assert_eq!(qwen.loaded_context_length, None, "not loaded");
        assert_eq!(qwen.context_length(), Some(262_144));
        assert_eq!(models[2].context_length(), None);
    }

    #[test]
    fn missing_fields_become_none_and_unknown_state_is_not_loaded() {
        let mystery = &fixture_models()[2];
        assert_eq!(mystery.arch, None);
        assert_eq!(mystery.quantization, None);
        assert_eq!(mystery.max_context_length, None);
        assert_eq!(mystery.loaded_context_length, None);
        assert!(mystery.capabilities.is_empty());
        assert_eq!(mystery.state, ModelState::NotLoaded);
    }

    #[test]
    fn tail_keeps_last_lines_and_splits_on_carriage_returns() {
        let text = "a\nb\r\n\rc 10%\rc 100%\n\nd\n";
        assert_eq!(tail(text, 3), "c 10%\nc 100%\nd");
        assert_eq!(tail("x", 5), "x");
    }

    /// Needs a running LM Studio server on the default port.
    #[tokio::test]
    #[ignore = "needs a live LM Studio server"]
    async fn live_list_models() {
        let lms = LmStudio::default();
        lms.ensure_server().await.unwrap();
        let models = lms.list_models().await.unwrap();
        println!("{} chat models:", models.len());
        for m in &models {
            println!("  {} {:?} {:?}", m.id, m.quantization, m.state);
        }
        assert!(!models.is_empty());
        assert!(models.iter().all(ModelInfo::is_chat));
    }

    /// Streams a short reply from whatever model is already loaded. Loads nothing.
    #[tokio::test]
    #[ignore = "needs a live LM Studio server with a model loaded"]
    async fn live_chat_stream() {
        let lms = LmStudio::default();
        let model = lms
            .loaded_model()
            .await
            .unwrap()
            .expect("load a model in LM Studio first");
        let messages = [Message::new(
            crate::Role::User,
            "Reply with exactly: hello from loops",
        )];
        let (tx, mut rx) = mpsc::channel(64);
        let started = Instant::now();
        let printer = tokio::spawn(async move {
            let mut reply = String::new();
            let mut usage = false;
            while let Some(delta) = rx.recv().await {
                println!("{:>6.0?} {delta:?}", started.elapsed());
                match delta {
                    ChatDelta::Content(text) => reply.push_str(&text),
                    ChatDelta::Usage { .. } => usage = true,
                    _ => {}
                }
            }
            (reply, usage)
        });
        lms.chat(&model.id, &messages, &[], tx).await.unwrap();
        let (reply, usage) = printer.await.unwrap();
        println!("{} replied {reply:?} in {:?}", model.id, started.elapsed());
        assert!(reply.to_lowercase().contains("hello from loops"));
        assert!(usage, "the usage chunk after `Done` got through");
    }

    /// Loads the smallest local model, checks it, then unloads everything.
    #[tokio::test]
    #[ignore = "loads a ~6GB model into LM Studio"]
    async fn live_load_and_unload() {
        let lms = LmStudio::default();
        lms.ensure_server().await.unwrap();
        let started = Instant::now();
        let model = lms.load_model("google/gemma-4-e4b").await.unwrap();
        println!("loaded {} in {:?}", model.id, started.elapsed());
        assert_eq!(model.state, ModelState::Loaded);
        lms.unload_all().await.unwrap();
        assert_eq!(lms.loaded_model().await.unwrap(), None);
    }

    /// Unloads `google/gemma-4-e4b`, which must already be loaded. Reload it
    /// afterwards with `lms load google/gemma-4-e4b -y`.
    #[tokio::test]
    #[ignore = "unloads google/gemma-4-e4b from a live LM Studio server"]
    async fn live_unload_model() {
        let lms = LmStudio::default();
        let started = Instant::now();
        lms.unload_model("google/gemma-4-e4b").await.unwrap();
        println!("unloaded google/gemma-4-e4b in {:?}", started.elapsed());
        assert_eq!(lms.loaded_model().await.unwrap(), None);
    }
}
