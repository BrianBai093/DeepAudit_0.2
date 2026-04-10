"""RAG layer for semantic code retrieval in DeepAudit."""

from p2c.rag.builder import build_code_index
from p2c.rag.index import CodeIndex
from p2c.rag.query import retrieve_for_claims

__all__ = ["CodeIndex", "build_code_index", "retrieve_for_claims"]
