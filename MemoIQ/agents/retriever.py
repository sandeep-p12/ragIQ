"""Retriever agent - tool-calling agent that calls RAG."""

import json
import logging
from typing import Any, Dict, List

try:
    from crewai import Agent
except ImportError as e:
    import sys
    error_msg = f"""crewai is required but not found. 

Install it with:
    python3 -m pip install "crewai>=0.80.0"

Current Python: {sys.executable}
Original error: {e}"""
    raise ImportError(error_msg) from e

# Try to import CrewAI BaseTool
try:
    from crewai.tools import BaseTool
    HAS_CREWAI_TOOL = True
except ImportError:
    HAS_CREWAI_TOOL = False

from MemoIQ.agents.base import create_agent
from MemoIQ.config import MemoIQConfig
from MemoIQ.rag.rag_adapter import rag_retrieve as rag_retrieve_func
from src.core.dataclasses import ContextPack

logger = logging.getLogger(__name__)


def create_retriever_agent(config: MemoIQConfig, all_chunks: List[Dict[str, Any]]) -> Agent:
    """
    Create retriever agent with RAG tool.
    
    Args:
        config: MemoIQConfig
        all_chunks: List of all chunks for neighbor expansion
        
    Returns:
        Agent with RAG retrieval tool
    """
    # Create RAG retrieval tool function
    def rag_retrieve_tool_func(query: str, doc_id: str) -> str:
        """
        Retrieve context from RAG system.
        
        Args:
            query: Search query
            doc_id: Document ID to search in
            
        Returns:
            JSON string with retrieved context and citations
        """
        try:
            result: ContextPack = rag_retrieve_func(
                query=query,
                doc_id=doc_id,
                config=config.retrieval_config,
                all_chunks=all_chunks,
                llm_config=config.agent_llm_config,
            )
            
            # Format result as JSON
            context_text = "\n\n".join([
                chunk.get("text_for_embedding", chunk.get("raw_md_fragment", str(chunk)))
                for chunk in result.selected_chunks[:10]
            ])
            
            response = {
                "query": query,
                "chunks": result.selected_chunks[:10],
                "citations": result.citations,
                "context_pack": context_text,
            }
            
            return json.dumps(response, indent=2, default=str)
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)
            return json.dumps({"error": str(e), "chunks": [], "citations": []})
    
    # Create CrewAI BaseTool from function
    if HAS_CREWAI_TOOL:
        class RAGRetrieveTool(BaseTool):
            name: str = "rag_retrieve"
            description: str = "Retrieve relevant context from indexed documents using RAG. Takes a query string and document ID, returns JSON with context, chunks, and citations."
            
            def _run(self, query: str, doc_id: str) -> str:
                return rag_retrieve_tool_func(query, doc_id)
        
        rag_retrieve_tool = RAGRetrieveTool()
    else:
        # Fallback: use function directly (may not work)
        logger.warning("CrewAI BaseTool not available, using function directly (may fail)")
        rag_retrieve_tool = rag_retrieve_tool_func
    
    return create_agent(
        name="retriever",
        role="Document Retriever",
        goal="Retrieve relevant context from indexed documents using RAG",
        backstory="""You are a Retriever agent specialized in retrieving relevant context from document indexes using RAG.
        When asked to retrieve information, you use the rag_retrieve tool to search indexed documents.
        You return retrieved context and citations in JSON format.
        You always include citations for all retrieved information.""",
        config=config,
        tools=[rag_retrieve_tool],
    )
