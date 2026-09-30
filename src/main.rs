mod catalog;
mod chat;
mod registry;
mod server;
mod tray;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

/// Runs local models for other programs, over a Unix socket.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the installed models. Never goes online.
    Serve {
        /// Socket to listen on
        #[arg(long, default_value_os_t = default_socket())]
        socket: PathBuf,
        /// Unload a model after this many minutes without a request
        #[arg(long, default_value_t = 5)]
        idle_minutes: u64,
    },
    /// Show a tray icon with what the service is doing
    Tray {
        /// Socket the service listens on
        #[arg(long, default_value_os_t = default_socket())]
        socket: PathBuf,
    },
    /// Download a model
    Pull { id: String },
    /// List the models this runtime knows and which are installed
    List,
}

fn default_socket() -> PathBuf {
    dirs::runtime_dir()
        .unwrap_or_else(catalog::data_dir)
        .join("model-runtime.sock")
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Serve {
            socket,
            idle_minutes,
        } => {
            gliner2_rs::init("model-runtime");
            tokio::runtime::Runtime::new()?.block_on(server::serve(
                socket,
                Duration::from_secs(idle_minutes * 60),
            ))
        }
        Command::Tray { socket } => tray::run(socket),
        Command::Pull { id } => {
            let Some(spec) = catalog::find(&id) else {
                bail!("no model named {id}");
            };
            let snapshot = spec.pull()?;
            println!("{id}: {}", snapshot.display());
            Ok(())
        }
        Command::List => {
            for spec in catalog::CATALOG {
                let state = if spec.installed() {
                    "installed"
                } else {
                    "not installed"
                };
                println!("{:<6} {:<14} {}", spec.id, state, spec.description);
            }
            Ok(())
        }
    }
}
