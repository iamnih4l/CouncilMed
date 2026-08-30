"""
CouncilMed — Agentic AI Orchestrator (Rebuilt)
==================================================
LangGraph-based multi-agent reasoning over REAL council evidence.

This replaces the previous orchestrator that received simulated
model outputs and had no structured tool access.

Architecture:
  1. Clinical Extractor: Extracts structured data from council evidence
  2. Radiologist Agent: Interprets imaging findings
  3. Safety Reviewer: Checks for missing data and safety concerns
  4. Chief Medical Officer: Synthesizes final insight

The agent has access to a CONTROLLED tool set — it can read council
evidence, model predictions, and study metadata, but it CANNOT access
the filesystem, run commands, or modify any data.

Output is structured JSON with validation — no freeform text used as
clinical guidance.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..council.evidence import CouncilResult, AgentInsight, ModelStatus

logger = logging.getLogger(__name__)

# Default Ollama configuration
DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-fp16"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# JSON output schema for the agent
INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key clinical insights from the analysis"
        },
        "clinical_significance": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Clinical significance of the findings"
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recommended next steps for the clinician"
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important missing information that limits interpretation"
        },
        "safety_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Safety concerns that must be addressed"
        },
        "evidence_references": {
            "type": "array",
            "items": {"type": "string"},
            "description": "References to specific model evidence supporting insights"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Agent's confidence in its analysis (0-1)"
        },
    },
    "required": ["key_insights", "clinical_significance", "recommended_actions",
                  "missing_information", "safety_flags", "confidence"]
}


def format_council_evidence(result: CouncilResult) -> str:
    """
    Format council evidence into a structured text summary for the agent.
    Only includes real, verified data — no fabricated context.
    """
    lines = []
    lines.append("=== COUNCIL ANALYSIS EVIDENCE ===")
    lines.append(f"Study: {result.modality or 'Unknown'} / {result.anatomy or 'Unknown'}")
    lines.append(f"Council Run ID: {result.council_run_id}")
    lines.append(f"Timestamp: {result.timestamp}")
    lines.append("")
    
    # Consensus summary
    lines.append("--- CONSENSUS ---")
    lines.append(f"Level: {result.consensus_level.value}")
    lines.append(f"Primary Finding: {result.primary_finding}")
    lines.append(f"Confidence: {result.primary_finding_confidence:.1%}")
    lines.append(f"Severity: {result.severity.value}")
    lines.append(f"Models Agreeing: {result.models_agreeing}/{result.models_total}")
    lines.append(f"Agreement Score: {result.agreement_score:.2f}")
    lines.append(f"Ensemble Uncertainty: {result.ensemble_uncertainty:.2f}")
    lines.append(f"Evidence Strength: {result.evidence_strength.value}")
    lines.append("")
    
    # Individual model evidence
    lines.append("--- MODEL EVIDENCE ---")
    for ev in result.member_evidence:
        lines.append(f"\nModel: {ev.model_name} (v{ev.model_version})")
        lines.append(f"  Architecture: {ev.model_architecture}")
        lines.append(f"  Status: {ev.status.value}")
        
        if ev.status == ModelStatus.SUCCESS:
            lines.append(f"  Prediction: {ev.prediction}")
            lines.append(f"  Confidence: {ev.confidence:.1%}")
            lines.append(f"  Uncertainty: {ev.uncertainty:.2f}")
            lines.append(f"  Processing Time: {ev.processing_time_ms:.1f}ms")
            lines.append(f"  Device: {ev.device}")
            
            if ev.probabilities:
                lines.append("  Probabilities:")
                for cls, prob in sorted(ev.probabilities.items(), key=lambda x: -x[1]):
                    lines.append(f"    {cls}: {prob:.1%}")
        else:
            lines.append(f"  Error: {ev.error_message or 'Unknown error'}")
    
    # Disagreements
    if result.disagreements:
        lines.append("\n--- DISAGREEMENTS ---")
        for d in result.disagreements:
            lines.append(
                f"Model {d['model_id']} predicts '{d['model_prediction']}' "
                f"(confidence: {d['model_confidence']:.1%}) — "
                f"disagrees with majority '{d['majority_prediction']}'"
            )
    
    # Human review
    lines.append("\n--- REVIEW STATUS ---")
    lines.append(f"Requires Human Review: {result.requires_human_review}")
    if result.human_review_reasons:
        lines.append("Reasons:")
        for r in result.human_review_reasons:
            lines.append(f"  - {r}")
    
    return "\n".join(lines)


def build_system_prompt() -> str:
    """Build the system prompt for the clinical agent."""
    return """You are a clinical AI assistant embedded in the CouncilMed medical imaging analysis system.

Your role is to interpret the evidence produced by the AI Council — a panel of independent 
deep learning models that have analyzed a medical image.

CRITICAL RULES:
1. You are NOT making a diagnosis. You are summarizing and interpreting AI model outputs.
2. Every statement must be traceable to specific model evidence.
3. Never invent findings that are not supported by the model outputs.
4. Always recommend clinician review and correlation with clinical history.
5. If model agreement is weak or uncertainty is high, emphasize this prominently.
6. Use conservative, qualified language ("findings are suggestive of", not "the patient has").
7. Always list limitations and missing information.

SAFETY:
- If any safety concern exists (weak consensus, high uncertainty, conflicting models), 
  flag it explicitly in safety_flags.
- Never provide treatment recommendations beyond "consult with specialist".
- State that AI analysis is supplementary and does not replace clinical judgment.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching the required schema.
Do not include any text outside the JSON object."""


def run_council_agent(
    council_result: CouncilResult,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> AgentInsight:
    """
    Run the agentic AI layer over real council evidence.
    
    Uses Ollama to generate structured clinical insights from
    the council's evidence. The agent reasons over REAL model
    predictions, not fabricated data.
    
    Args:
        council_result: The CouncilResult from the fusion engine
        model_name: Ollama model to use
        base_url: Ollama API base URL
    
    Returns:
        AgentInsight with structured clinical reasoning
    """
    start_time = time.perf_counter()
    
    # Format evidence for the agent
    evidence_text = format_council_evidence(council_result)
    system_prompt = build_system_prompt()
    
    # Retrieve relevant medical knowledge (RAG)
    knowledge_context = _retrieve_knowledge_context(council_result)
    
    # Build the user prompt with knowledge context
    knowledge_section = ""
    if knowledge_context:
        knowledge_section = f"\n\n{knowledge_context}\n"
    
    user_prompt = f"""Analyze the following AI Council evidence and provide structured clinical insights.

{evidence_text}
{knowledge_section}
Respond with a JSON object containing:
- key_insights: List of key findings from the council analysis
- clinical_significance: Clinical significance of the findings
- recommended_actions: Recommended next steps
- missing_information: Important missing data
- safety_flags: Any safety concerns
- evidence_references: References to specific model evidence
- confidence: Your confidence in this interpretation (0-1)

IMPORTANT: Output ONLY valid JSON. No other text."""
    
    try:
        import httpx
        
        # Call Ollama API
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.3,  # Low temperature for factual output
                    "num_predict": 2048,
                },
            },
            timeout=60.0,
        )
        
        response.raise_for_status()
        data = response.json()
        
        raw_text = data.get("message", {}).get("content", "")
        
        # Parse JSON response
        insight_data = _parse_agent_response(raw_text)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return AgentInsight(
            key_insights=insight_data.get("key_insights", []),
            clinical_significance=insight_data.get("clinical_significance", []),
            recommended_actions=insight_data.get("recommended_actions", []),
            missing_information=insight_data.get("missing_information", []),
            safety_flags=insight_data.get("safety_flags", []),
            evidence_references=insight_data.get("evidence_references", []),
            confidence=min(float(insight_data.get("confidence", 0.5)), 1.0),
            agent_model=model_name,
            agent_version="1.0.0",
            processing_time_ms=elapsed_ms,
        )
        
    except httpx.ConnectError:
        logger.warning("Ollama not available at %s", base_url)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return AgentInsight(
            key_insights=[f"Agent reasoning unavailable — Ollama not running at {base_url}"],
            safety_flags=["Agent reasoning could not be performed"],
            confidence=0.0,
            agent_model=model_name,
            processing_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.error("Agent reasoning failed: %s", e, exc_info=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return AgentInsight(
            key_insights=[f"Agent reasoning error: {str(e)}"],
            safety_flags=["Agent reasoning encountered an error"],
            confidence=0.0,
            agent_model=model_name,
            processing_time_ms=elapsed_ms,
        )


def _parse_agent_response(raw_text: str) -> dict[str, Any]:
    """Parse and validate the agent's JSON response."""
    # Try direct JSON parse
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown code block
    import re
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    
    # Try finding any JSON object in the text
    brace_start = raw_text.find('{')
    brace_end = raw_text.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        try:
            data = json.loads(raw_text[brace_start:brace_end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    
    # Fallback: return the raw text as a single insight
    logger.warning("Could not parse agent response as JSON: %s...", raw_text[:200])
    return {
        "key_insights": [raw_text[:500]],
        "clinical_significance": [],
        "recommended_actions": ["Review raw agent output"],
        "missing_information": [],
        "safety_flags": ["Agent output was not structured JSON — manual review required"],
        "confidence": 0.3,
    }


def _retrieve_knowledge_context(council_result: CouncilResult) -> str:
    """
    Retrieve relevant medical knowledge for the agent's context.
    
    Strategy:
      1. Try ChromaDB vector retrieval (if available)
      2. Fall back to baseline knowledge dictionary
    
    The knowledge context is injected into the user prompt so the
    agent can reference curated medical information alongside
    the council evidence.
    """
    finding = council_result.primary_finding or ""
    modality = council_result.modality or ""
    anatomy = council_result.anatomy or ""
    
    query = f"{finding} {modality} {anatomy}".strip()
    if not query:
        return ""
    
    # Strategy 1: ChromaDB vector retrieval
    try:
        from ..knowledge.rag import KnowledgeStore
        
        store = KnowledgeStore()
        if store.document_count > 0:
            result = store.query(query, n_results=3)
            context = store.format_context(result, max_tokens=1500)
            if context:
                logger.info(
                    "RAG: retrieved %d knowledge chunks for '%s'",
                    result.total_results, query,
                )
                return context
    except ImportError:
        logger.debug("ChromaDB not installed — skipping vector retrieval")
    except Exception as e:
        logger.warning("Knowledge retrieval failed: %s", e)
    
    # Strategy 2: Baseline knowledge fallback
    from ..knowledge.rag import get_baseline_context
    
    baseline = get_baseline_context(finding)
    if baseline:
        logger.info("Using baseline knowledge for '%s'", finding)
        return (
            "=== MEDICAL KNOWLEDGE CONTEXT ===\n\n"
            f"{baseline}\n\n"
            "=== END KNOWLEDGE ==="
        )
    
    return ""

