//! The models this runtime knows, and where they live on disk.

use std::path::PathBuf;

use anyhow::{Context, Result};
use gliner2_rs::{Precision, hub};

pub struct Spec {
    pub id: &'static str,
    pub description: &'static str,
    pub source: Source,
}

pub enum Source {
    /// A GLiNER2 model from gliner2-rs's list, run to find spans.
    Gliner(hub::Model),
    /// One GGUF file in a Hugging Face repository, run by llama.cpp to chat.
    Gguf {
        repo: &'static str,
        file: &'static str,
    },
}

pub const CATALOG: &[Spec] = &[
    Spec {
        id: "pii",
        description: "GLiNER2-PII: 42 PII types in EN, FR, ES, DE, IT, PT, NL",
        source: Source::Gliner(hub::PRIVACY_PII_MULTI),
    },
    Spec {
        id: "gemma",
        description: "Gemma 4 E2B, instruction-tuned, 4-bit QAT: chat",
        source: Source::Gguf {
            repo: "google/gemma-4-E2B-it-qat-q4_0-gguf",
            file: "gemma-4-E2B_q4_0-it.gguf",
        },
    },
];

pub fn find(id: &str) -> Option<&'static Spec> {
    CATALOG.iter().find(|s| s.id == id)
}

pub fn data_dir() -> PathBuf {
    dirs::data_dir()
        .expect("no data directory on this system")
        .join("model-runtime")
}

impl Spec {
    /// A link to the model's snapshot in the Hugging Face cache, made by `pull`.
    pub fn dir(&self) -> PathBuf {
        data_dir().join("models").join(self.id)
    }

    pub fn installed(&self) -> bool {
        self.dir().exists()
    }

    /// Downloads the model. The only thing in this program that goes online.
    pub fn pull(&self) -> Result<PathBuf> {
        let snapshot = match self.source {
            Source::Gliner(model) => hub::download(model, Precision::Fp32)?.0,
            Source::Gguf { repo, file } => {
                let file = hf_hub::api::sync::Api::new()?
                    .model(repo.into())
                    .get(file)?;
                file.parent().unwrap().to_path_buf()
            }
        };
        let link = self.dir();
        std::fs::create_dir_all(link.parent().unwrap())?;
        if link.symlink_metadata().is_ok() {
            std::fs::remove_file(&link)?;
        }
        std::os::unix::fs::symlink(&snapshot, &link)
            .with_context(|| format!("linking {} to {}", link.display(), snapshot.display()))?;
        Ok(snapshot)
    }
}
