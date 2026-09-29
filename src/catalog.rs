//! The models this runtime knows, and where they live on disk.

use std::path::PathBuf;

use anyhow::{Context, Result};
use gliner2_rs::{Precision, hub};

pub struct Spec {
    pub id: &'static str,
    pub description: &'static str,
    pub hub: hub::Model,
}

pub const CATALOG: &[Spec] = &[Spec {
    id: "pii",
    description: "GLiNER2-PII: 42 PII types in EN, FR, ES, DE, IT, PT, NL",
    hub: hub::PRIVACY_PII_MULTI,
}];

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
        let (snapshot, _) = hub::download(self.hub, Precision::Fp32)?;
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
