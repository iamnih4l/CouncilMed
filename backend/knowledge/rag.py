"""
CouncilMed — Knowledge RAG Engine
=====================================
Retrieval-Augmented Generation for medical knowledge.

This module provides the AI council's agent with curated medical
knowledge context. It uses a local vector store (ChromaDB) to
index and retrieve relevant medical literature, imaging guidelines,
and differential diagnosis information.

Architecture:
  1. Ingestion: Markdown/PDF → chunked → embedded → stored in ChromaDB
  2. Retrieval: Query → embedded → nearest-neighbor search → top-k chunks
  3. Augmentation: Chunks injected into the agent's system prompt

All processing is LOCAL. No data leaves the machine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ==============================
# Data Classes
# ==============================

@dataclass
class KnowledgeChunk:
    """A single chunk of knowledge from a document."""
    chunk_id: str
    content: str
    source: str  # File path or URL
    title: str = ""
    section: str = ""
    category: str = ""  # e.g., "radiology", "pathology", "protocol"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class RetrievalResult:
    """Result from a knowledge retrieval query."""
    chunks: list[KnowledgeChunk]
    query: str
    distances: list[float]
    total_results: int


# ==============================
# Text Chunking
# ==============================

def chunk_markdown(
    text: str,
    source: str,
    max_chunk_size: int = 512,
    overlap: int = 64,
) -> list[KnowledgeChunk]:
    """
    Split markdown text into overlapping chunks for embedding.
    
    Uses section-aware splitting — tries to break at headers and
    paragraphs rather than mid-sentence.
    """
    chunks: list[KnowledgeChunk] = []
    
    # Split by headers first
    sections = re.split(r'\n(#{1,4}\s+.+)\n', text)
    
    current_section = ""
    current_text = ""
    
    for part in sections:
        if re.match(r'^#{1,4}\s+', part):
            current_section = part.strip('# ').strip()
            continue
        
        # Split by paragraphs
        paragraphs = part.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_text) + len(para) > max_chunk_size:
                if current_text:
                    chunk_id = hashlib.md5(
                        f"{source}:{len(chunks)}:{current_text[:32]}".encode()
                    ).hexdigest()[:12]
                    
                    chunks.append(KnowledgeChunk(
                        chunk_id=chunk_id,
                        content=current_text.strip(),
                        source=source,
                        section=current_section,
                    ))
                    
                    # Keep overlap
                    words = current_text.split()
                    overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else ''
                    current_text = overlap_text + '\n\n' + para
                else:
                    current_text = para
            else:
                current_text += '\n\n' + para if current_text else para
    
    # Final chunk
    if current_text.strip():
        chunk_id = hashlib.md5(
            f"{source}:{len(chunks)}:{current_text[:32]}".encode()
        ).hexdigest()[:12]
        
        chunks.append(KnowledgeChunk(
            chunk_id=chunk_id,
            content=current_text.strip(),
            source=source,
            section=current_section,
        ))
    
    return chunks


# ==============================
# Vector Store (ChromaDB)
# ==============================

class KnowledgeStore:
    """
    Local vector store for medical knowledge retrieval.
    
    Uses ChromaDB for persistence and sentence-transformers for
    embedding. All processing happens locally.
    """
    
    def __init__(
        self,
        persist_dir: str | Path = "./data/knowledge_db",
        collection_name: str = "councilmed_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._client = None
        self._collection = None
        self._embedding_fn = None
    
    def _ensure_initialized(self) -> None:
        """Lazy-initialize ChromaDB and embedding model."""
        if self._client is not None:
            return
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "ChromaDB is required for knowledge RAG. "
                "Install with: pip install chromadb"
            )
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )
        
        # Use sentence-transformers for local embedding
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(
            "Knowledge store initialized: %s (%d documents)",
            self.persist_dir,
            self._collection.count(),
        )
    
    def ingest_file(self, file_path: str | Path) -> int:
        """
        Ingest a markdown file into the knowledge store.
        Returns the number of chunks added.
        """
        self._ensure_initialized()
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {path}")
        
        text = path.read_text(encoding='utf-8')
        chunks = chunk_markdown(text, source=str(path))
        
        if not chunks:
            return 0
        
        # Add to ChromaDB
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "section": c.section,
                "category": c.category,
                "content_hash": c.content_hash,
            }
            for c in chunks
        ]
        
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        
        logger.info("Ingested %d chunks from %s", len(chunks), path.name)
        return len(chunks)
    
    def ingest_directory(self, directory: str | Path) -> int:
        """Ingest all .md files from a directory."""
        dir_path = Path(directory)
        total = 0
        
        for md_file in sorted(dir_path.glob("**/*.md")):
            try:
                count = self.ingest_file(md_file)
                total += count
            except Exception as e:
                logger.error("Failed to ingest %s: %s", md_file, e)
        
        return total
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        category_filter: str | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge chunks for a question.
        
        Returns the top-k most semantically similar chunks.
        """
        self._ensure_initialized()
        
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        results = self._collection.query(
            query_texts=[question],
            n_results=min(n_results, self._collection.count() or 1),
            where=where_filter,
        )
        
        chunks = []
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        
        for i, (doc, meta, chunk_id) in enumerate(zip(documents, metadatas, ids)):
            chunks.append(KnowledgeChunk(
                chunk_id=chunk_id,
                content=doc,
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                category=meta.get("category", ""),
            ))
        
        return RetrievalResult(
            chunks=chunks,
            query=question,
            distances=distances,
            total_results=len(chunks),
        )
    
    def format_context(
        self,
        retrieval: RetrievalResult,
        max_tokens: int = 2048,
    ) -> str:
        """
        Format retrieved chunks as context for the agent prompt.
        
        Respects a token budget to avoid exceeding context limits.
        """
        context_parts: list[str] = []
        estimated_tokens = 0
        
        for chunk in retrieval.chunks:
            # Rough token estimate: ~4 chars per token
            chunk_tokens = len(chunk.content) // 4
            
            if estimated_tokens + chunk_tokens > max_tokens:
                break
            
            header = f"[Source: {chunk.source}"
            if chunk.section:
                header += f" | Section: {chunk.section}"
            header += "]"
            
            context_parts.append(f"{header}\n{chunk.content}")
            estimated_tokens += chunk_tokens
        
        if not context_parts:
            return ""
        
        return (
            "=== RETRIEVED MEDICAL KNOWLEDGE ===\n\n"
            + "\n\n---\n\n".join(context_parts)
            + "\n\n=== END KNOWLEDGE ==="
        )
    
    @property
    def document_count(self) -> int:
        """Number of chunks in the store."""
        self._ensure_initialized()
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all knowledge from the store."""
        self._ensure_initialized()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )


# ==============================
# Curated Medical Knowledge
# ==============================

# Seed knowledge for common findings — used when ChromaDB
# is not available or for baseline context
BASELINE_KNOWLEDGE: dict[str, str] = {
    "glioblastoma": (
        "Glioblastoma (GBM, WHO Grade IV) is the most aggressive primary brain tumor. "
        "On MRI, GBM typically shows heterogeneous enhancement with central necrosis, "
        "surrounding edema, and mass effect. T1-weighted post-contrast images reveal "
        "irregular ring enhancement. T2/FLAIR shows peritumoral edema extending beyond "
        "the enhancing margin. DWI may show restricted diffusion at the tumor margin. "
        "Median survival is approximately 15 months with standard treatment."
    ),
    "meningioma": (
        "Meningioma is the most common primary intracranial tumor, arising from "
        "meningothelial cells. On MRI, meningiomas are typically well-circumscribed, "
        "extra-axial masses with homogeneous enhancement on T1 post-contrast. A dural "
        "tail sign (linear enhancement along adjacent dura) is characteristic but not "
        "pathognomonic. Most meningiomas are WHO Grade I (benign) and slow-growing."
    ),
    "pituitary_adenoma": (
        "Pituitary adenomas are classified as microadenomas (<10mm) or macroadenomas "
        "(≥10mm). On MRI, they typically appear as sellar masses that are iso/hypointense "
        "on T1 and hyperintense on T2. Dynamic contrast-enhanced MRI shows delayed "
        "enhancement compared to normal pituitary tissue. Macroadenomas may cause mass "
        "effect on the optic chiasm (bitemporal hemianopia) or cavernous sinus invasion."
    ),
    "normal_brain": (
        "Normal brain MRI shows symmetric gray and white matter with preserved sulcal "
        "pattern and ventricular morphology. T1-weighted images show gray matter as "
        "intermediate signal and white matter as bright. T2-weighted images show CSF "
        "as bright, gray matter as intermediate, and white matter as dark. No areas "
        "of abnormal enhancement, restricted diffusion, or mass effect should be present."
    ),
    "pneumonia": (
        "Chest X-ray findings of pneumonia include airspace opacification (consolidation), "
        "which may be lobar, segmental, or patchy. Air bronchograms may be visible within "
        "the consolidation. Associated findings include pleural effusion, atelectasis, and "
        "lymphadenopathy. CT shows ground-glass opacities, consolidation, and possible "
        "tree-in-bud pattern in infectious bronchiolitis."
    ),
    "fracture": (
        "Fractures appear as cortical discontinuity on radiographs. Key features include "
        "fracture line orientation, number of fragments, displacement, angulation, and "
        "joint involvement. Associated soft tissue swelling is an important secondary sign. "
        "Occult fractures may require CT or MRI for detection, particularly in the spine, "
        "pelvis, and small bones."
    ),
}


def get_baseline_context(finding: str) -> str:
    """
    Get baseline medical context for a finding.
    Used as fallback when ChromaDB is not available.
    """
    finding_lower = finding.lower().replace(' ', '_')
    
    # Try exact match
    if finding_lower in BASELINE_KNOWLEDGE:
        return BASELINE_KNOWLEDGE[finding_lower]
    
    # Try partial match
    for key, text in BASELINE_KNOWLEDGE.items():
        if key in finding_lower or finding_lower in key:
            return text
    
    return ""
