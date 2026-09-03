import sys
import os
from pathlib import Path

def get_base_dir() -> Path:
    """
    Get the base directory of the application.
    When running in development, it resolves to the root of the project.
    When running as a PyInstaller executable (Tauri sidecar), it uses the current working directory,
    which Tauri sets to the app's root/installation directory.
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller executable
        cwd = Path.cwd()
        if (cwd / "data").exists() or (cwd / "model_registry").exists():
            return cwd
        else:
            # Fallback for sidecars that might not have CWD set perfectly
            return Path(sys.executable).parent
    else:
        # Development mode
        return Path(__file__).parent.parent
