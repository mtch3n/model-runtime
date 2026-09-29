//! HTTP over a Unix socket.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::{UnixListener, UnixStream};

use crate::registry::{ModelInfo, Registry};

type Shared = Arc<Registry>;

pub async fn serve(socket: PathBuf, idle: Duration) -> Result<()> {
    let registry: Shared = Arc::new(Registry::new(idle));
    let listener = bind(&socket).await?;
    eprintln!("listening on {}", socket.display());

    let app = Router::new()
        .route("/models", get(models))
        .route("/models/{id}/load", post(load))
        .route("/models/{id}/unload", post(unload))
        .route("/models/{id}/detect", post(detect))
        .with_state(registry);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    std::fs::remove_file(&socket)?;
    Ok(())
}

/// Binds the socket so only this user can connect: the texts sent to it are
/// the private data it's looking for.
async fn bind(socket: &Path) -> Result<UnixListener> {
    if UnixStream::connect(socket).await.is_ok() {
        bail!("model-runtime is already running on {}", socket.display());
    }
    if socket.exists() {
        std::fs::remove_file(socket)?;
    }
    std::fs::create_dir_all(socket.parent().unwrap())?;
    let listener =
        UnixListener::bind(socket).with_context(|| format!("binding {}", socket.display()))?;
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

async fn shutdown() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[derive(Serialize)]
struct ModelList {
    models: Vec<ModelInfo>,
    /// Memory the whole runtime is using, loaded models included.
    rss_mb: Option<u64>,
}

async fn models(State(registry): State<Shared>) -> Json<ModelList> {
    Json(ModelList {
        models: registry.list(),
        rss_mb: rss_mb(),
    })
}

fn rss_mb() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096 / 1_000_000)
}

async fn load(
    State(registry): State<Shared>,
    UrlPath(id): UrlPath<String>,
) -> Result<StatusCode, Error> {
    registry.load(&id).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn unload(
    State(registry): State<Shared>,
    UrlPath(id): UrlPath<String>,
) -> Result<StatusCode, Error> {
    registry.unload(&id).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct DetectRequest {
    fields: Vec<Field>,
    /// Types to look for, in the model's words, like `email` or `iban`.
    /// Every type it knows when left out.
    labels: Option<Vec<String>>,
    #[serde(default = "default_threshold")]
    threshold: f32,
}

fn default_threshold() -> f32 {
    0.5
}

#[derive(Deserialize)]
struct Field {
    field_id: String,
    text: String,
}

#[derive(Serialize)]
struct DetectResponse {
    fields: Vec<FieldSpans>,
}

#[derive(Serialize)]
struct FieldSpans {
    field_id: String,
    spans: Vec<SpanOut>,
}

/// `start` and `end` are byte offsets into the field's text, end exclusive.
#[derive(Serialize)]
struct SpanOut {
    start: usize,
    end: usize,
    #[serde(rename = "type")]
    kind: String,
    score: f32,
}

async fn detect(
    State(registry): State<Shared>,
    UrlPath(id): UrlPath<String>,
    Json(req): Json<DetectRequest>,
) -> Result<Json<DetectResponse>, Error> {
    let (ids, texts): (Vec<String>, Vec<String>) =
        req.fields.into_iter().map(|f| (f.field_id, f.text)).unzip();
    let spans = registry
        .detect(&id, texts, req.labels, req.threshold)
        .await?;
    let fields = ids
        .into_iter()
        .zip(spans)
        .map(|(field_id, spans)| FieldSpans {
            field_id,
            spans: spans
                .into_iter()
                .map(|s| SpanOut {
                    start: s.start,
                    end: s.end,
                    kind: s.label,
                    score: s.score,
                })
                .collect(),
        })
        .collect();
    Ok(Json(DetectResponse { fields }))
}

struct Error(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for Error {
    fn from(e: E) -> Error {
        Error(e.into())
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let body = Json(json!({ "error": format!("{:#}", self.0) }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}
