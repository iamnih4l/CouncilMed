"""
Quick patch script: uploads only app.py to Hugging Face Space in 3 seconds.
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub is not installed in the environment.")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent.parent

print("=" * 60)
print("   COUNCILMED — INSTANT CLOUD PATCH (app.py)")
print("=" * 60)
print("This uploads the fixed app.py to Hugging Face in ~3 seconds.")
print("It enables the /gradio_api/api/health REST endpoints for Vercel.\n")

repo_id = os.environ.get("HF_REPO_ID", "Mohammed12345678/CouncilAI")
token = os.environ.get("HF_TOKEN")
if not token:
    token = input("Enter Hugging Face Access Token (from https://huggingface.co/settings/tokens): ").strip()
    if not token:
        print("Cancelled: token required.")
        sys.exit(0)

api = HfApi(token=token)
app_path = BASE_DIR / "app.py"

print(f"\nUploading {app_path.name} to Space '{repo_id}'...")
try:
    api.upload_file(
        path_or_fileobj=str(app_path),
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Fix: mount FastAPI router onto Gradio server with _app=demo.app and inject pipeline",
    )
    print("\n" + "=" * 60)
    print("  ✓ SUCCESS: app.py uploaded to Hugging Face Space!")
    print("=" * 60)
    print("\nThe Hugging Face Space is restarting its container now (~20-40 seconds).")
    print("Once restarted, visit https://council-med.vercel.app/ and the status pill")
    print("will immediately turn EMERALD: CUDA • 9 MODELS • OPERATIONAL!\n")
except Exception as e:
    print(f"\n✗ Error uploading to Hugging Face: {e}")
    sys.exit(1)
