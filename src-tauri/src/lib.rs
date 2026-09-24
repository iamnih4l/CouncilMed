// CouncilMed — Tauri Desktop Application
// ==========================================
// Manages the Python backend lifecycle and provides
// IPC commands for the React frontend.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

use std::sync::Mutex;
use tauri::{Manager, State};

use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};

// ==============================
// State
// ==============================

struct AppState {
    backend_process: Mutex<Option<CommandChild>>,
    backend_port: u16,
}

// ==============================
// IPC Commands
// ==============================

/// Start the Python backend process
#[tauri::command]
fn start_backend(app: tauri::AppHandle, state: State<AppState>) -> Result<String, String> {
    let mut process_guard = state.backend_process.lock().map_err(|e| e.to_string())?;

    if process_guard.is_some() {
        return Ok("Backend already running".to_string());
    }

    let (mut rx, child) = app
        .shell()
        .sidecar("app-backend")
        .map_err(|e| format!("Failed to create sidecar: {}", e))?
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let CommandEvent::Stdout(line) = event {
                println!("backend: {}", String::from_utf8_lossy(&line));
            } else if let CommandEvent::Stderr(line) = event {
                eprintln!("backend err: {}", String::from_utf8_lossy(&line));
            }
        }
    });

    *process_guard = Some(child);

    Ok(format!("Backend started on port {}", state.backend_port))
}

/// Stop the Python backend process
#[tauri::command]
fn stop_backend(state: State<AppState>) -> Result<String, String> {
    let mut process_guard = state.backend_process.lock().map_err(|e| e.to_string())?;

    if let Some(child) = process_guard.take() {
        child.kill().map_err(|e| format!("Failed to stop backend: {}", e))?;
        Ok("Backend stopped".to_string())
    } else {
        Ok("Backend was not running".to_string())
    }
}

/// Check if the Python backend is running
#[tauri::command]
fn check_backend(state: State<AppState>) -> Result<bool, String> {
    let process_guard = state.backend_process.lock().map_err(|e| e.to_string())?;
    Ok(process_guard.is_some())
}

/// Get the backend API URL
#[tauri::command]
fn get_backend_url(state: State<AppState>) -> String {
    format!("http://127.0.0.1:{}", state.backend_port)
}

/// Detect available hardware (GPU, RAM, etc.)
#[tauri::command]
fn detect_hardware() -> Result<HardwareInfo, String> {
    Ok(HardwareInfo {
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        // GPU detection would use system-specific calls
        gpu_available: false,
        gpu_name: None,
    })
}

#[derive(Serialize, Deserialize)]
struct HardwareInfo {
    platform: String,
    arch: String,
    gpu_available: bool,
    gpu_name: Option<String>,
}

// ==============================
// Entry Point
// ==============================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            backend_process: Mutex::new(None),
            backend_port: 8000,
        })
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            use tauri::Manager;
            let (mut rx, child) = app
                .shell()
                .sidecar("app-backend")
                .map_err(|e| e.to_string())?
                .spawn()
                .map_err(|e| e.to_string())?;

            let binding = app.state::<AppState>();
            let mut process_guard = binding.backend_process.lock().unwrap();
            *process_guard = Some(child);

            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line) = event {
                        println!("backend: {}", String::from_utf8_lossy(&line));
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            check_backend,
            get_backend_url,
            detect_hardware,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Stop backend when window is closed
                let state: State<AppState> = window.state();
                let _ = stop_backend(state);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running CouncilMed");
}
