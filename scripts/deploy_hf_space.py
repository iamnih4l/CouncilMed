"""
CouncilMed — Hugging Face Spaces 1-Click Deployment Script
==========================================================
Uploads the CouncilMed backend, model weights, and Dockerfile
to your Hugging Face Space (Free 16 GB RAM CPU Tier).

Usage:
    python scripts/deploy_hf_space.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    print("=" * 65)
    print("   COUNCILMED — HUGGING FACE SPACES CLOUD DEPLOYER")
    print("=" * 65)
    print("\nThis script deploys the CouncilMed backend to a FREE 24/7 cloud instance")
    print("on Hugging Face Spaces (16 GB RAM, 2 vCPUs, Docker SDK).\n")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is not installed in the environment.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # 1. Ask for Hugging Face Username / Repo
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        repo_input = input("Enter your Hugging Face Space ID [default: Mohammed12345678/CouncilAI]: ").strip()
        repo_id = repo_input if repo_input else "Mohammed12345678/CouncilAI"

    # 2. Ask for Hugging Face Token (with WRITE permission)
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_input = input("Enter your Hugging Face Access Token (from https://huggingface.co/settings/tokens): ").strip()
        if not token_input:
            print("Operation cancelled: HF Token is required.")
            sys.exit(0)
        token = token_input

    api = HfApi(token=token)

    # 3. Ensure Space exists
    print(f"\n[1/3] Verifying Space '{repo_id}' on Hugging Face...")
    space_ready = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print("  ✓ Space exists!")
        space_ready = True
    except Exception:
        print(f"  Space '{repo_id}' not found online yet.")
        print("  Attempting to create it with free Gradio SDK...")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="gradio",
                private=False,
                token=token,
            )
            print("  ✓ Created new Gradio Space successfully!")
            space_ready = True
        except Exception as e:
            print(f"\n  ⚠️ Auto-creation notice: {e}")
            print("\n  Hugging Face requires creating the Space once on their web interface.")
            print("  Please do this quick 10-second setup:")
            print("  -------------------------------------------------------------")
            print("  1. Open: https://huggingface.co/new-space")
            print(f"  2. Space Name: {repo_id.split('/')[-1]}")
            print("  3. Space SDK:  Select 'Gradio'")
            print("  4. Hardware:   Select 'ZeroGPU' or 'CPU Basic'")
            print("  5. Click:      'Create Space'")
            print("  -------------------------------------------------------------")
            input("\n  Once you click 'Create Space' on the website, press [ENTER] here to continue upload...")
            
            # Re-verify
            try:
                api.repo_info(repo_id=repo_id, repo_type="space")
                print("  ✓ Space verified successfully!")
                space_ready = True
            except Exception as err:
                print(f"  ✗ Still could not find Space '{repo_id}': {err}")
                print("  Please ensure the Space name matches exactly and retry.")
                sys.exit(1)

    # 4. Upload files
    print("\n[2/3] Uploading CouncilMed application, model weights, and Gradio engine...")
    print("      (Target payload: ~299 MB across 77 files, well within the 1 GB Space limit)")

    # Explicit allow patterns: ONLY upload what the backend and models need.
    # Excludes large unrelated directories like src-tauri (3.5 GB), venv, node_modules, etc.
    allow_patterns = [
        "app.py",
        "README.md",
        "requirements.txt",
        "backend/**",
        "model_registry/**",
        "eva-data/**",
    ]

    ignore_patterns = [
        "**/__pycache__/**",
        "*.pyc",
        "*.pyo",
    ]

    try:
        commit_info = api.upload_folder(
            folder_path=str(BASE_DIR),
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            commit_message="Deploy CouncilMed AI Workstation (299 MB core backend + models)",
        )
        print("  ✓ Upload complete!")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        sys.exit(1)

    # 5. Output Results
    space_user, space_name = repo_id.split("/")
    direct_backend_url = f"https://{space_user.replace('_', '-')}-{space_name.replace('_', '-')}.hf.space"
    space_web_url = f"https://huggingface.co/spaces/{repo_id}"

    print("\n[3/3] Deployment initiated successfully! 🎉")
    print("=" * 65)
    print(f"Hugging Face Space:  {space_web_url}")
    print(f"Permanent Cloud URL: {direct_backend_url}")
    print("=" * 65)
    print("\nNEXT STEPS FOR VERCEL:")
    print("1. Hugging Face will now build the container (takes ~2 minutes).")
    print(f"   Check build logs at: {space_web_url}")
    print(f"2. Once status turns GREEN ('Running'), go to Vercel -> Project Settings -> Environment Variables.")
    print("3. Set or update:")
    print(f"     VITE_API_URL = {direct_backend_url}")
    print("4. Click 'Redeploy' on Vercel.")
    print("\nNow your application will work 24/7 from ANY device or laptop with ZERO local setup!")
    print("=" * 65)


if __name__ == "__main__":
    main()
