#!/usr/bin/env python3
"""
CouncilMed Autonomous Quality Assurance (QA) Agent — Entrypoint
==============================================================
Runs an end-to-end technical, clinical-safety, functional, and performance
audit of the CouncilMed clinical diagnostic platform.

Usage:
    python run_qa_audit.py
    python run_qa_audit.py --no-fix
    python run_qa_audit.py --help
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa_agent.runner import QAAgentRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CouncilMed Autonomous Medical AI Systems QA Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Skip applying automated remediation fixes before running audits.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="audit_results.json",
        help="Path where structured audit results JSON will be written.",
    )
    parser.add_argument(
        "--md-out",
        type=str,
        default="AUDIT_REPORT.md",
        help="Path where comprehensive Markdown audit report will be written.",
    )

    args = parser.parse_args()

    print("Initializing CouncilMed QA Agent...")
    runner = QAAgentRunner(root_dir=PROJECT_ROOT, auto_fix=not args.no_fix)
    report = runner.run_full_audit()

    # Verify report status
    overall = report.final_readiness.get("OVERALL_PROJECT_READINESS", "UNKNOWN")
    print(f"\nAudit completed. Status: {overall}")
    print(f"Results JSON: {PROJECT_ROOT / args.json_out}")
    print(f"Report Markdown: {PROJECT_ROOT / args.md_out}")

    return 0 if overall == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
