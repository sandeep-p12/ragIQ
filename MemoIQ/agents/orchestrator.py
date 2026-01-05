"""MemoIQ orchestrator - main coordination system."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import RateLimitError, APIError
except ImportError:
    # Fallback if openai is not available
    RateLimitError = Exception
    APIError = Exception

from MemoIQ.agents.consistency_checker import (
    check_consistency,
    create_consistency_checker_agent,
)
from MemoIQ.agents.coordinator import WorkflowState, create_coordinator_agent
from MemoIQ.agents.extract_covenants import (
    create_covenant_extractor_agent,
    extract_covenant_field,
)
from MemoIQ.agents.extract_financials import (
    create_financial_extractor_agent,
    extract_financial_field,
)
from MemoIQ.agents.human_proxy import create_human_proxy_agent
from MemoIQ.agents.memo_writer import create_memo_writer_agent
from MemoIQ.agents.policy_checker import check_policy, create_policy_checker_agent
from MemoIQ.agents.retriever import create_retriever_agent
from MemoIQ.agents.synthesize_risks import create_risk_synthesizer_agent
from MemoIQ.agents.template_analyzer import analyze_template
from MemoIQ.config import MemoIQConfig
from MemoIQ.rag.doc_ingest import ingest_reference_documents
from MemoIQ.rag.rag_adapter import rag_retrieve as rag_retrieve_func
from MemoIQ.schema import EvidencePack, FieldExtraction, MemoDraft, TemplateSchema
from MemoIQ.template.template_filler import TemplateFiller
from MemoIQ.utils.io import create_run_directory, save_draft, save_json
from src.core.dataclasses import ContextPack

logger = logging.getLogger(__name__)


class MemoIQOrchestrator:
    """Main orchestrator for MemoIQ system."""
    
    def __init__(self, config: MemoIQConfig):
        """Initialize orchestrator."""
        self.config = config
        self.template_filler = TemplateFiller(config)
    
    def run(
        self,
        template_path: str,
        reference_doc_paths: List[str],
        run_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run complete MemoIQ workflow.
        
        Args:
            template_path: Path to template file
            reference_doc_paths: List of reference document paths
            run_dir: Optional run directory (creates new if None)
            progress_callback: Optional callback function(stage: str, progress: float) for progress updates
            
        Returns:
            Dict with results
        """
        # Create run directory
        if run_dir is None:
            run_dir = create_run_directory(self.config)
        
        logger.info(f"Starting MemoIQ run in {run_dir}")
        
        def update_progress(stage: str, progress: float):
            """Helper to update progress if callback provided."""
            if progress_callback:
                try:
                    progress_callback(stage, progress)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
        
        try:
            # Step 1: Analyze template
            update_progress("Analyzing template", 0.1)
            logger.info("Step 1: Analyzing template...")
            template_schema = analyze_template(template_path, self.config)
            save_json(run_dir, "template_schema.json", template_schema.model_dump())
            
            # Step 2: Ingest reference documents
            update_progress("Ingesting reference documents", 0.2)
            logger.info("Step 2: Ingesting reference documents...")
            # Use strategy from config if available, otherwise default to AUTO
            from src.config.parsing_strategies import StrategyEnum
            strategy = getattr(self.config, 'parsing_strategy', None) or StrategyEnum.AUTO
            logger.info(f"Using parsing strategy: {strategy} (from config.parsing_strategy: {getattr(self.config, 'parsing_strategy', None)})")
            doc_results = ingest_reference_documents(
                reference_doc_paths,
                self.config.parsing_config,
                self.config.chunking_config,
                self.config.retrieval_config,
                index_documents=True,
                strategy=strategy,
            )
            
            # Collect all chunks for retrieval
            all_chunks = []
            for doc_id, (children, parents, stats) in doc_results.items():
                all_chunks.extend([c.__dict__ for c in children])
                all_chunks.extend([p.__dict__ for p in parents])
            
            # Step 3: Initialize agents (for potential future use with CrewAI tasks)
            update_progress("Initializing agents", 0.4)
            logger.info("Step 3: Initializing agents...")
            coordinator = create_coordinator_agent(self.config)
            retriever = create_retriever_agent(self.config, all_chunks)
            financial_extractor = create_financial_extractor_agent(self.config)
            covenant_extractor = create_covenant_extractor_agent(self.config)
            risk_synthesizer = create_risk_synthesizer_agent(self.config)
            memo_writer = create_memo_writer_agent(self.config)
            consistency_checker = create_consistency_checker_agent(self.config)
            policy_checker = create_policy_checker_agent(self.config)
            human_proxy = create_human_proxy_agent(self.config)
            
            # Step 4: Create workflow state
            update_progress("Creating extraction plan", 0.5)
            workflow_state = WorkflowState(template_schema)
            extraction_plan = workflow_state.create_extraction_plan()
            
            # Step 5: Extract fields
            update_progress("Extracting fields", 0.6)
            logger.info("Step 5: Extracting fields...")
            
            # Get all unique doc_ids from chunks for searching across all documents
            all_doc_ids = set()
            if all_chunks:
                for chunk in all_chunks:
                    doc_id = chunk.get("doc_id")
                    if doc_id and doc_id != "unknown":
                        all_doc_ids.add(doc_id)
            if doc_results:
                all_doc_ids.update(doc_results.keys())
            
            logger.info(f"Searching across {len(all_doc_ids)} reference documents: {list(all_doc_ids)}")
            
            # Track errors for summary
            rate_limit_errors = []
            other_errors = []
            
            for field_id, agent_name in extraction_plan.items():
                field_def = template_schema.field_definitions[field_id]
                
                # Retrieve context - search across ALL reference documents
                query = f"{field_def.name}: {field_def.description or ''}"
                logger.info(f"Extracting field '{field_id}' ({field_def.name}) using {agent_name}")
                
                try:
                    # Search across all documents by not filtering by doc_id
                    # This allows retrieval from all reference documents
                    context_pack: ContextPack = rag_retrieve_func(
                        query=query,
                        doc_id=None,  # None means search all documents
                        config=self.config.retrieval_config,
                        all_chunks=all_chunks,
                        llm_config=self.config.agent_llm_config,
                    )
                    
                    if not context_pack.selected_chunks:
                        logger.warning(f"No chunks found for field '{field_id}'. Trying individual documents...")
                        # Fallback: try searching each document individually
                        all_context_chunks = []
                        for doc_id in all_doc_ids:
                            try:
                                doc_context = rag_retrieve_func(
                                    query=query,
                                    doc_id=doc_id,
                                    config=self.config.retrieval_config,
                                    all_chunks=all_chunks,
                                    llm_config=self.config.agent_llm_config,
                                )
                                all_context_chunks.extend(doc_context.selected_chunks[:3])  # Top 3 per doc
                            except (RateLimitError, APIError) as e:
                                # Check if it's a rate limit error
                                error_str = str(e).lower()
                                if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                                    logger.warning(f"⚠️  Rate limit error retrieving from doc_id {doc_id}: {e}")
                                    raise  # Re-raise to be caught by outer handler
                                else:
                                    logger.warning(f"API error retrieving from doc_id {doc_id}: {e}")
                            except Exception as e:
                                logger.warning(f"Error retrieving from doc_id {doc_id}: {e}")
                        
                        if all_context_chunks:
                            # Use combined chunks from all documents
                            context_pack.selected_chunks = all_context_chunks[:10]  # Top 10 overall
                            logger.info(f"Found {len(context_pack.selected_chunks)} chunks across all documents")
                    
                    # Build context text from selected chunks (use more chunks for better context)
                    context_text = "\n\n".join([
                        chunk.get("text_for_embedding", chunk.get("raw_md_fragment", str(chunk)))
                        for chunk in context_pack.selected_chunks[:10]  # Use top 10 chunks for better context
                    ])
                    
                    if not context_text.strip():
                        logger.warning(f"Empty context for field '{field_id}'. Field extraction may be inaccurate.")
                    
                    # Extract field based on agent type (now using config directly)
                    if agent_name == "financial_extractor":
                        extraction = extract_financial_field(
                            field_def.name,
                            context_text,
                            self.config,  # Pass config instead of agent
                        )
                    else:
                        extraction = extract_covenant_field(
                            field_def.name,
                            context_text,
                            self.config,  # Pass config instead of agent
                        )
                    
                    # Ensure field_id and field_type are set correctly
                    extraction_dict = extraction.model_dump()
                    extraction_dict["field_id"] = field_id
                    extraction_dict["field_type"] = field_def.field_type
                    extraction = FieldExtraction(**extraction_dict)
                    
                    workflow_state.add_extraction(field_id, extraction)
                    logger.info(f"✓ Extracted {field_id}: {extraction.value} (confidence: {extraction.confidence:.2f})")
                    if extraction.citations:
                        logger.info(f"  Citations: {len(extraction.citations)} sources")
                    
                except (RateLimitError, APIError) as e:
                    # Check if it's a rate limit error
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                        rate_limit_errors.append(field_id)
                        logger.error(f"⚠️  Rate limit error extracting {field_id} ({field_def.name}): {e}")
                        logger.error(f"   This is likely due to OpenAI API quota being exceeded. Please check your billing.")
                        logger.error(f"   The workflow will continue with other fields, but this field will be empty.")
                    else:
                        other_errors.append(field_id)
                        logger.error(f"✗ API error extracting {field_id}: {e}")
                    
                    # Add empty extraction so workflow can continue
                    workflow_state.add_extraction(field_id, FieldExtraction(
                        field_id=field_id,
                        value=None,
                        confidence=0.0,
                        citations=[],
                        field_type=field_def.field_type,
                    ))
                    # Add a small delay to avoid hammering the API
                    if rate_limit_errors:
                        time.sleep(0.5)
                    
                except Exception as e:
                    other_errors.append(field_id)
                    logger.error(f"✗ Error extracting {field_id}: {e}", exc_info=True)
                    # Add empty extraction so workflow can continue
                    workflow_state.add_extraction(field_id, FieldExtraction(
                        field_id=field_id,
                        value=None,
                        confidence=0.0,
                        citations=[],
                        field_type=field_def.field_type,
                    ))
            
            # Log summary of errors
            if rate_limit_errors or other_errors:
                logger.warning("\n" + "=" * 80)
                logger.warning("Field Extraction Summary")
                logger.warning("=" * 80)
                if rate_limit_errors:
                    logger.warning(f"⚠️  Rate limit errors: {len(rate_limit_errors)} fields failed due to OpenAI quota/rate limits")
                    logger.warning(f"   Affected fields: {', '.join(rate_limit_errors[:10])}{'...' if len(rate_limit_errors) > 10 else ''}")
                    logger.warning(f"   Action: Check your OpenAI account billing and quota limits")
                if other_errors:
                    logger.warning(f"✗ Other errors: {len(other_errors)} fields failed due to other issues")
                    logger.warning(f"   Affected fields: {', '.join(other_errors[:10])}{'...' if len(other_errors) > 10 else ''}")
                logger.warning("=" * 80 + "\n")
            
            # Step 6: Validate
            update_progress("Validating", 0.8)
            logger.info("Step 6: Validating...")
            consistency_records = check_consistency(
                workflow_state.extracted_fields,
                self.config,  # Pass config instead of agent
            )
            for record in consistency_records:
                workflow_state.add_validation(record)
            
            # Policy validation (optional - only if policy_rules.json exists and has rules)
            policy_rules_path = Path("MemoIQ/policy/policy_rules.json")
            if policy_rules_path.exists():
                logger.info("Running policy validation...")
                policy_records = check_policy(
                    workflow_state.extracted_fields,
                    str(policy_rules_path),
                    self.config,  # Pass config instead of agent
                )
                if policy_records:
                    logger.info(f"Policy validation found {len(policy_records)} issues")
                    for record in policy_records:
                        workflow_state.add_validation(record)
                else:
                    logger.info("Policy validation passed (no issues found)")
            else:
                logger.info(f"Policy rules file not found at {policy_rules_path}. Skipping policy validation.")
                logger.info("Template will be filled based on reference documents only.")
            
            # Step 7: Fill template
            update_progress("Filling template", 0.9)
            logger.info("Step 7: Filling template...")
            draft_path = run_dir / "outputs" / "draft_v1.docx"
            filled_path = self.template_filler.fill_template(
                template_schema,
                workflow_state.extracted_fields,
                str(draft_path),
            )
            
            # Step 8: Create evidence pack
            evidence_pack = {}
            for field_id, extraction in workflow_state.extracted_fields.items():
                evidence_pack[field_id] = EvidencePack(
                    field_id=field_id,
                    value=extraction.value,
                    citations=extraction.citations,
                    extraction_context=extraction.raw_context,
                    confidence=extraction.confidence,
                )
            
            # Step 9: Create memo draft
            memo_draft = MemoDraft(
                draft_docx_path=str(filled_path),
                extracted_fields=workflow_state.extracted_fields,
                validation_report=workflow_state.validation_records,
                evidence_pack=evidence_pack,
                structure_preserved=True,
                draft_version=1,
            )
            
            # Save results
            save_json(run_dir, "extracted_fields.json", {
                k: v.model_dump() for k, v in workflow_state.extracted_fields.items()
            })
            save_json(run_dir, "validation_report.json", [
                v.model_dump() for v in workflow_state.validation_records
            ])
            save_json(run_dir, "evidence_pack.json", {
                k: v.model_dump() for k, v in evidence_pack.items()
            })
            
            logger.info(f"MemoIQ run complete. Draft saved to {filled_path}")
            
            return {
                "run_dir": str(run_dir),
                "memo_draft": memo_draft,
                "template_schema": template_schema,
                "workflow_state": workflow_state,
            }
            
        except Exception as e:
            logger.error(f"Error in MemoIQ run: {e}", exc_info=True)
            raise
