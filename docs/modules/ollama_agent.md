# Ollama Agent Module

The Ollama Agent module (`backend/agents/orchestrator.py`) provides an interpretive intelligence layer over the mathematical outputs of the PyTorch models.

## 1. What is this module?
It is a LangGraph/httpx-based interface connecting the CouncilMed backend to a locally running instance of Ollama (defaulting to `llama3:8b-instruct-fp16`).

## 2. Why does it exist?
While PyTorch models provide statistical probabilities (e.g., "Meningioma: 92%"), doctors need clinical context. They need to know the *significance* of that finding, what safety flags to watch for, and what actions to consider. The agent translates math into medicine.

## 3. Strict Separation of Concerns
**CRITICAL**: The LLM does *not* look at the image. It does *not* make the primary diagnosis. The LLM only receives the text-based `CouncilResult` (the mathematical consensus) and is forced via system prompts to explain that specific result. This prevents the LLM from hallucinating diagnoses that disagree with the dedicated PyTorch vision models.

## 4. The Workflow
1. **Context Preparation**: The `CouncilResult` is passed to `format_council_evidence()`, which builds a strict text summary of the consensus, model confidence, and disagreements.
2. **Knowledge Retrieval (RAG)**: `_retrieve_knowledge_context()` takes the primary finding (e.g., "Meningioma") and queries the ChromaDB vector store (`backend/knowledge/rag.py`).
3. **Prompt Construction**: The formatted evidence and the retrieved clinical guidelines are combined into a system prompt.
4. **Execution**: The prompt is sent to `http://localhost:11434/api/chat`.
5. **JSON Enforcement**: The prompt mandates output matching the `INSIGHT_SCHEMA`. The parser `_parse_agent_response()` extracts this JSON.

## 5. Output Format
The module returns an `AgentInsight` object containing:
- `key_insights`
- `clinical_significance`
- `recommended_actions`
- `missing_information`
- `safety_flags`

## 6. Failure Behavior
If Ollama is not running or the network request fails, the module gracefully catches the `httpx.ConnectError` and returns an `AgentInsight` stating that "Agent reasoning unavailable". The primary PyTorch inference still succeeds, ensuring the system does not crash just because the LLM is offline.
