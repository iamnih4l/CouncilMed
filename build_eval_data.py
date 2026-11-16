#!/usr/bin/env python3
"""
CouncilMed — Real Clinical Evaluation Dataset Generator
=======================================================
Downloads authentic, peer-reviewed clinical benchmark scans for X-Ray, CT, and MRI
directly into the 'eva-data' evaluation subfolders.
"""

from pathlib import Path
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add scripts directory to path and delegate to clinical populator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.populate_real_eva_data import populate_all

if __name__ == "__main__":
    populate_all()
