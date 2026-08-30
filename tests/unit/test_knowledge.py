"""
CouncilMed — Knowledge RAG Unit Tests
=========================================
Tests for text chunking, baseline knowledge, and vector store.
ChromaDB-dependent tests are skipped if not installed.
"""

import pytest

from backend.knowledge.rag import (
    chunk_markdown,
    KnowledgeChunk,
    get_baseline_context,
    BASELINE_KNOWLEDGE,
)


# ==============================
# Chunking Tests
# ==============================

class TestChunkMarkdown:
    def test_empty_text_produces_no_chunks(self):
        chunks = chunk_markdown("", source="test.md")
        assert len(chunks) == 0
    
    def test_whitespace_only_produces_no_chunks(self):
        chunks = chunk_markdown("   \n\n   ", source="test.md")
        assert len(chunks) == 0
    
    def test_single_paragraph_produces_one_chunk(self):
        text = "This is a short paragraph about brain MRI findings."
        chunks = chunk_markdown(text, source="test.md")
        assert len(chunks) == 1
        assert "brain MRI" in chunks[0].content
    
    def test_chunk_has_source(self):
        text = "Test content for source tracking."
        chunks = chunk_markdown(text, source="brain_mri.md")
        assert chunks[0].source == "brain_mri.md"
    
    def test_long_text_produces_multiple_chunks(self):
        # Create text longer than max_chunk_size
        paragraphs = [f"Paragraph {i}: " + "x" * 200 for i in range(10)]
        text = "\n\n".join(paragraphs)
        
        chunks = chunk_markdown(text, source="test.md", max_chunk_size=300)
        assert len(chunks) > 1
    
    def test_chunks_have_unique_ids(self):
        paragraphs = [f"Paragraph {i}: " + "x" * 200 for i in range(5)]
        text = "\n\n".join(paragraphs)
        
        chunks = chunk_markdown(text, source="test.md", max_chunk_size=300)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
    
    def test_section_headers_captured(self):
        text = """# Brain Tumors

This is intro text about brain tumors.

## Glioblastoma

This is a paragraph about glioblastoma.

## Meningioma

This is a paragraph about meningioma.
"""
        chunks = chunk_markdown(text, source="test.md")
        # At least one chunk should have a non-empty section
        has_section = any(c.section != "" for c in chunks)
        assert has_section
    
    def test_content_hash_is_deterministic(self):
        text = "Deterministic content for hashing."
        chunks1 = chunk_markdown(text, source="test.md")
        chunks2 = chunk_markdown(text, source="test.md")
        
        assert chunks1[0].content_hash == chunks2[0].content_hash
    
    def test_content_hash_changes_with_content(self):
        chunks1 = chunk_markdown("Content A", source="test.md")
        chunks2 = chunk_markdown("Content B", source="test.md")
        
        assert chunks1[0].content_hash != chunks2[0].content_hash


# ==============================
# Baseline Knowledge Tests
# ==============================

class TestBaselineKnowledge:
    def test_glioblastoma_knowledge_exists(self):
        assert "glioblastoma" in BASELINE_KNOWLEDGE
    
    def test_meningioma_knowledge_exists(self):
        assert "meningioma" in BASELINE_KNOWLEDGE
    
    def test_normal_brain_knowledge_exists(self):
        assert "normal_brain" in BASELINE_KNOWLEDGE
    
    def test_get_baseline_exact_match(self):
        context = get_baseline_context("glioblastoma")
        assert "WHO Grade IV" in context
    
    def test_get_baseline_partial_match(self):
        context = get_baseline_context("suspected glioblastoma")
        assert len(context) > 0
    
    def test_get_baseline_no_match(self):
        context = get_baseline_context("completely_unknown_finding_xyz")
        assert context == ""
    
    def test_get_baseline_case_insensitive(self):
        context = get_baseline_context("Meningioma")
        assert "arachnoid" in context.lower() or "meningioma" in context.lower()
    
    def test_all_baseline_entries_are_nonempty(self):
        for key, text in BASELINE_KNOWLEDGE.items():
            assert len(text) > 50, f"Baseline entry '{key}' is too short"


# ==============================
# Knowledge Store Tests (ChromaDB required)
# ==============================

class TestKnowledgeStore:
    """Tests that require ChromaDB — skipped if not installed."""
    
    @pytest.fixture
    def store(self, tmp_path):
        chromadb = pytest.importorskip("chromadb", reason="ChromaDB not installed")
        from backend.knowledge.rag import KnowledgeStore
        return KnowledgeStore(
            persist_dir=str(tmp_path / "test_knowledge_db"),
            collection_name="test_collection",
        )
    
    def test_store_initializes(self, store):
        assert store.document_count == 0
    
    def test_ingest_and_query(self, store, tmp_path):
        # Write test knowledge
        test_file = tmp_path / "test_knowledge.md"
        test_file.write_text(
            "# Brain Tumors\n\n"
            "Glioblastoma is the most aggressive brain tumor. "
            "It shows ring enhancement on MRI.\n\n"
            "## Meningioma\n\n"
            "Meningiomas are typically benign extra-axial tumors."
        )
        
        count = store.ingest_file(test_file)
        assert count > 0
        assert store.document_count > 0
        
        # Query
        result = store.query("What does glioblastoma look like on MRI?")
        assert result.total_results > 0
        assert any("glioblastoma" in c.content.lower() for c in result.chunks)
    
    def test_format_context(self, store, tmp_path):
        test_file = tmp_path / "test.md"
        test_file.write_text("Glioblastoma shows ring enhancement on T1 post-contrast MRI.")
        
        store.ingest_file(test_file)
        result = store.query("glioblastoma MRI")
        
        context = store.format_context(result)
        assert "RETRIEVED MEDICAL KNOWLEDGE" in context
        assert "END KNOWLEDGE" in context
    
    def test_clear_store(self, store, tmp_path):
        test_file = tmp_path / "test.md"
        test_file.write_text("Test content for clearing.")
        
        store.ingest_file(test_file)
        assert store.document_count > 0
        
        store.clear()
        assert store.document_count == 0
