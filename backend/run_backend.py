import sys
import os
import multiprocessing

# Add the project root to sys.path to allow imports from "backend" to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.api.app import app
import uvicorn

if __name__ == "__main__":
    # Required for PyInstaller multi-processing on Windows
    multiprocessing.freeze_support()
    
    # Run the Uvicorn server programmatically
    # We use app directly instead of a string to avoid import issues in PyInstaller
    uvicorn.run(app, host="127.0.0.1", port=8000)
