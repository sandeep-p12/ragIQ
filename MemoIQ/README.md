# MemoIQ - Credit Memo Drafting Automation

MemoIQ is a multi-agent system built on CrewAI that automates credit memo drafting by orchestrating document parsing, information extraction, validation, and template filling.

## Architecture Overview

MemoIQ uses a multi-agent architecture with the following components:

1. **Template Analyzer**: Analyzes PDF/DOCX templates to detect fillable fields (empty cells, underscore lines, checkboxes)
2. **RAG Adapter**: Thin wrappers around existing RAG IQ pipeline for parsing, chunking, indexing, and retrieval
3. **Specialist Agents**: 
   - **Retriever**: Tool-calling agent that retrieves context from RAG system
   - **Financial Extractor**: Extracts financial metrics
   - **Covenant Extractor**: Extracts covenant terms
   - **Risk Synthesizer**: Synthesizes risk factors
   - **Memo Writer**: Writes narrative sections
4. **Validators**:
   - **Consistency Checker**: Validates consistency across fields
   - **Policy Checker**: Validates against credit policy rules
5. **Coordinator**: Orchestrates workflow and manages state
6. **Human Proxy**: Human-in-the-loop for review and revision

## Key Features

- **No Explicit Placeholders**: Templates use empty cells, underscore lines, and checkboxes - no `{{field_name}}` syntax required
- **Intelligent Field Detection**: Fields are inferred from template structure and context
- **Structure Preservation**: Filled drafts match original template structure exactly
- **Always DOCX Output**: All drafts are downloadable DOCX files (PDF templates converted to DOCX)
- **Citation Tracking**: All extracted values include citations to source documents
- **Validation**: Automatic consistency and policy validation
- **Human Review**: Built-in human feedback and revision workflow

## Usage

### Via Streamlit UI

1. Navigate to the "MemoIQ" tab in the Streamlit app
2. Upload a template file (PDF or DOCX)
3. Upload reference documents (PDF, DOCX, XLSX, CSV, MD, HTML, TXT)
4. Click "Run MemoIQ"
5. Review results, download draft, provide feedback

### Via Python API

```python
from MemoIQ.agents.orchestrator import MemoIQOrchestrator
from MemoIQ.config import MemoIQConfig

config = MemoIQConfig()
orchestrator = MemoIQOrchestrator(config)

results = orchestrator.run(
    template_path="template.pdf",
    reference_doc_paths=["doc1.pdf", "doc2.docx"],
)

draft = results["memo_draft"]
print(f"Draft saved to: {draft.draft_docx_path}")
```

## Workflow

1. **Template Analysis**: Parse template and detect fillable fields
2. **Document Ingestion**: Parse, chunk, and index reference documents
3. **Field Extraction**: Use specialist agents to extract field values with RAG
4. **Validation**: Check consistency and policy compliance
5. **Template Filling**: Fill template with extracted values while preserving structure
6. **Human Review**: Present draft, validation report, and evidence pack for review
7. **Revision**: Update draft based on feedback (future enhancement)

## File Structure

```
MemoIQ/
├── config.py              # MemoIQ configuration
├── schema.py              # Pydantic models
├── template/
│   ├── template_parser.py    # Parse templates, detect fields
│   └── template_filler.py    # Fill templates with values
├── rag/
│   ├── rag_adapter.py        # Thin wrappers around /src
│   └── doc_ingest.py          # Document ingestion orchestration
├── agents/
│   ├── base.py                # Base CrewAI configuration
│   ├── orchestrator.py        # Main orchestrator
│   ├── coordinator.py         # Workflow coordinator
│   ├── template_analyzer.py   # Template analysis agent
│   ├── retriever.py           # RAG retrieval agent
│   ├── extract_financials.py  # Financial extraction agent
│   ├── extract_covenants.py   # Covenant extraction agent
│   ├── synthesize_risks.py    # Risk synthesis agent
│   ├── memo_writer.py         # Memo writing agent
│   ├── consistency_checker.py # Consistency validation agent
│   ├── policy_checker.py      # Policy validation agent
│   └── human_proxy.py         # Human-in-the-loop agent
├── prompts/
│   ├── field_prompts.py       # Field extraction prompts
│   └── section_prompts.py     # Section writing prompts
├── policy/
│   └── policy_rules.json      # Credit policy validation rules
└── utils/
    ├── io.py                  # Run management
    ├── citations.py           # Citation utilities
    ├── diff.py                # Draft comparison
    └── metrics.py             # Quality metrics
```

## Configuration

MemoIQ reuses configuration from `/src`:
- `ParseForgeConfig` for parsing and LLM
- `ChunkConfig` for chunking
- `RetrievalConfig` for retrieval

MemoIQ-specific settings in `MemoIQConfig`:
- `runs_dir`: Directory for storing run artifacts
- `agent_temperature`: LLM temperature for agents
- `agent_max_turns`: Maximum conversation turns

## Run Storage

Each run creates a directory under `MemoIQ/runs/{run_id}/` with:
- `template/`: Template file
- `reference_docs/`: Reference documents
- `template_schema.json`: Parsed template schema
- `extracted_fields.json`: All extracted fields
- `outputs/draft_v1.docx`: First filled draft
- `validation_report.json`: Validation results
- `evidence_pack.json`: Field → citations mapping
- `delta_log.json`: Changes between versions (if revised)

## Dependencies

- `crewai>=0.80.0`: Multi-agent framework
- `langchain-openai>=0.2.0`: LLM integration for CrewAI
- All dependencies from `/src` (parsing, chunking, retrieval)

## Notes

- **Maximum Reuse**: All RAG logic comes from `/src` - no duplication
- **Structure Preservation**: Templates are filled in place, maintaining all formatting
- **Citation Tracking**: All extractions include source citations
- **Human-in-Loop**: Final authority always with human reviewer

