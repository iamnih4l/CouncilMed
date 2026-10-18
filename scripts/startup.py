#!/usr/bin/env python
"""
CouncilMed — System Startup Script
=====================================
Validates the environment, initializes the database,
and starts the development servers.

Usage:
    python scripts/startup.py           # Full startup
    python scripts/startup.py --check   # Validate environment only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


REQUIRED_PYTHON_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "torch",
    "numpy",
    "PIL",
    "pydicom",
    "yaml",
]

OPTIONAL_PACKAGES = {
    "chromadb": "Knowledge RAG (retrieval-augmented generation)",
    "langchain_ollama": "Agentic AI reasoning",
    "langgraph": "Agent orchestration",
    "onnxruntime": "ONNX model inference",
    "tensorflow": "Keras model inference",
    "monai": "Medical image transforms",
}

# Colors for terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header():
    print(f"""
{CYAN}╔══════════════════════════════════════════╗
║        CouncilMed Startup Script         ║
║     Medical AI Workstation v0.1.0        ║
╚══════════════════════════════════════════╝{RESET}
""")


def check_python():
    version = sys.version_info
    print(f"  Python: {version.major}.{version.minor}.{version.micro}", end="")
    if version.major == 3 and version.minor >= 10:
        print(f" {GREEN}✓{RESET}")
        return True
    else:
        print(f" {RED}✗ (requires >= 3.10){RESET}")
        return False


def check_node():
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=5
        )
        version = result.stdout.strip()
        print(f"  Node.js: {version} {GREEN}✓{RESET}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  Node.js: {RED}✗ Not found{RESET}")
        return False


def check_npm():
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True, text=True, timeout=5,
            shell=True
        )
        version = result.stdout.strip()
        print(f"  npm: {version} {GREEN}✓{RESET}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  npm: {RED}✗ Not found{RESET}")
        return False


def check_rust():
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        version = result.stdout.strip()
        print(f"  Rust: {version} {GREEN}✓{RESET}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  Rust: {YELLOW}○ Not installed (optional for desktop){RESET}")
        return False


def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            model_count = max(0, len(lines) - 1)  # Subtract header
            print(f"  Ollama: {model_count} models available {GREEN}✓{RESET}")
            return True
        else:
            print(f"  Ollama: {YELLOW}○ Running but no models{RESET}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  Ollama: {YELLOW}○ Not installed (optional for agent){RESET}")
        return False


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  GPU: {name} ({mem:.1f} GB) {GREEN}✓{RESET}")
            return True
        else:
            print(f"  GPU: {YELLOW}○ CPU only (CUDA not available){RESET}")
            return False
    except ImportError:
        print(f"  GPU: {YELLOW}○ PyTorch not installed{RESET}")
        return False


def check_packages():
    missing_required = []
    for pkg in REQUIRED_PYTHON_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)
    
    if missing_required:
        print(f"  Required packages: {RED}✗ Missing: {', '.join(missing_required)}{RESET}")
        return False
    else:
        print(f"  Required packages: All {len(REQUIRED_PYTHON_PACKAGES)} present {GREEN}✓{RESET}")
    
    available_optional = []
    missing_optional = []
    for pkg, desc in OPTIONAL_PACKAGES.items():
        try:
            __import__(pkg)
            available_optional.append(pkg)
        except ImportError:
            missing_optional.append((pkg, desc))
    
    if available_optional:
        print(f"  Optional packages: {len(available_optional)} available {GREEN}✓{RESET}")
    if missing_optional:
        for pkg, desc in missing_optional:
            print(f"    {YELLOW}○ {pkg}: {desc}{RESET}")
    
    return True


def check_model_registry():
    registry_path = Path("model_registry")
    if not registry_path.exists():
        print(f"  Model Registry: {RED}✗ Directory not found{RESET}")
        return False
    
    manifests = list(registry_path.glob("**/*.yaml")) + list(registry_path.glob("**/*.yml"))
    print(f"  Model Registry: {len(manifests)} manifests found {GREEN}✓{RESET}")
    
    # Check for installed weights
    for manifest in manifests:
        import yaml
        with open(manifest) as f:
            config = yaml.safe_load(f)
        
        weight_path = config.get("weights_path", "")
        if weight_path and Path(weight_path).exists():
            print(f"    {GREEN}✓{RESET} {config.get('name', manifest.name)} (weights found)")
        else:
            print(f"    {YELLOW}○{RESET} {config.get('name', manifest.name)} (weights not found)")
    
    return True


def check_node_modules():
    if Path("node_modules").exists():
        print(f"  Node modules: Installed {GREEN}✓{RESET}")
        return True
    else:
        print(f"  Node modules: {RED}✗ Not installed (run: npm install){RESET}")
        return False


def check_knowledge():
    knowledge_path = Path("data/knowledge")
    if knowledge_path.exists():
        md_files = list(knowledge_path.glob("*.md"))
        print(f"  Knowledge base: {len(md_files)} files {GREEN}✓{RESET}")
    else:
        print(f"  Knowledge base: {YELLOW}○ Not found{RESET}")


def run_checks():
    print(f"{BOLD}Environment Checks{RESET}")
    print("─" * 40)
    
    all_ok = True
    
    all_ok &= check_python()
    all_ok &= check_node()
    all_ok &= check_npm()
    check_rust()  # Optional
    check_ollama()  # Optional
    check_gpu()  # Optional
    
    print()
    print(f"{BOLD}Dependencies{RESET}")
    print("─" * 40)
    
    all_ok &= check_packages()
    all_ok &= check_node_modules()
    
    print()
    print(f"{BOLD}Project{RESET}")
    print("─" * 40)
    
    check_model_registry()
    check_knowledge()
    
    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="CouncilMed Startup Script")
    parser.add_argument("--check", action="store_true", help="Validate environment only")
    args = parser.parse_args()
    
    print_header()
    
    ok = run_checks()
    
    if args.check:
        if ok:
            print(f"{GREEN}✓ All required checks passed.{RESET}")
        else:
            print(f"{RED}✗ Some required checks failed. Fix the issues above.{RESET}")
        sys.exit(0 if ok else 1)
    
    if not ok:
        print(f"{RED}Cannot start: fix required dependencies first.{RESET}")
        print(f"  pip install -r backend/requirements.txt")
        print(f"  npm install")
        sys.exit(1)
    
    print(f"{GREEN}Starting CouncilMed...{RESET}")
    print()
    
    # Start backend
    print(f"{CYAN}Starting Python backend on port 8000...{RESET}")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.app:app", "--reload", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    print(f"{CYAN}Starting Vite frontend on port 5173...{RESET}")
    frontend = subprocess.Popen(
        ["npm", "run", "dev:frontend"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
    )
    
    print()
    print(f"{GREEN}╔══════════════════════════════════════════╗{RESET}")
    print(f"{GREEN}║  CouncilMed is starting!                 ║{RESET}")
    print(f"{GREEN}║  Frontend:  http://localhost:5173         ║{RESET}")
    print(f"{GREEN}║  Backend:   http://localhost:8000         ║{RESET}")
    print(f"{GREEN}║  API Docs:  http://localhost:8000/docs    ║{RESET}")
    print(f"{GREEN}╚══════════════════════════════════════════╝{RESET}")
    print()
    print("Press Ctrl+C to stop.")
    
    try:
        backend.wait()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down...{RESET}")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print(f"{GREEN}CouncilMed stopped.{RESET}")


if __name__ == "__main__":
    main()
