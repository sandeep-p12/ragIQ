# MemoIQ - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Agents](#agents)
4. [Prompts](#prompts)
5. [Tools and Integrations](#tools-and-integrations)
6. [Workflow](#workflow)
7. [Template System](#template-system)
8. [RAG Integration](#rag-integration)
9. [Validation System](#validation-system)
10. [Configuration](#configuration)
11. [Data Models](#data-models)
12. [Utilities](#utilities)
13. [Run Management](#run-management)

---

## Overview

MemoIQ is a multi-agent system built on CrewAI that automates credit memo drafting. It orchestrates document parsing, information extraction, validation, and template filling using a sophisticated RAG (Retrieval-Augmented Generation) pipeline.

### Key Features

- **No Explicit Placeholders**: Templates use empty cells, underscore lines, and checkboxes - no `{{field_name}}` syntax required
- **Intelligent Field Detection**: Fields are inferred from template structure and context
- **Structure Preservation**: Filled drafts match original template structure exactly
- **Always DOCX Output**: All drafts are downloadable DOCX files (PDF templates converted to DOCX)
- **Citation Tracking**: All extracted values include citations to source documents
- **Validation**: Automatic consistency and policy validation
- **Human Review**: Built-in human feedback and revision workflow

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoIQ Orchestrator                        │
│  (Main workflow coordinator and state manager)                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Template   │   │   Document   │   │   Field      │
│   Analyzer   │   │   Ingestion  │   │  Extraction  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        │                   ▼                   │
        │          ┌──────────────┐            │
        │          │  RAG System  │            │
        │          │  (ParseForge)│            │
        │          └──────────────┘            │
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Template    │   │  Validation  │   │   Template   │
│   Filler     │   │   System     │   │   Output     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Component Layers

1. **Orchestration Layer**: `MemoIQOrchestrator` - Main workflow coordinator
2. **Agent Layer**: Specialized CrewAI agents for different tasks
3. **RAG Layer**: Document parsing, chunking, indexing, and retrieval
4. **Template Layer**: Template parsing and filling
5. **Validation Layer**: Consistency and policy checking
6. **Storage Layer**: Run management and artifact storage

### Technology Stack

- **Multi-Agent Framework**: CrewAI (v0.80.0+)
- **LLM Integration**: LangChain OpenAI/Azure OpenAI
- **Document Processing**: ParseForge (from `/src`)
- **Template Processing**: python-docx
- **Vector Store**: Pinecone (via `/src/storage/vector/pinecone.py`)
- **Configuration**: Pydantic models

---

## Agents

MemoIQ uses specialized CrewAI agents, each with a specific role and set of capabilities.

### 1. Template Analyzer Agent

**File**: `agents/template_analyzer.py`

**Role**: Analyzes template documents and extracts field definitions

**Responsibilities**:
- Parse PDF/DOCX templates
- Detect fillable fields (empty cells, underscore lines, checkboxes)
- Infer field meanings from context (section headers, nearby labels)
- Create structured `TemplateSchema` with field definitions

**Key Function**: `analyze_template(template_path, config) -> TemplateSchema`

**Implementation Details**:
- Uses `TemplateParser` class for actual parsing
- Supports both PDF and DOCX formats
- For PDFs: Converts to markdown first, then detects structure
- For DOCX: Directly parses tables and paragraphs
- Field detection strategies:
  - **Empty cells**: Cells with no text or only whitespace
  - **Underscore lines**: Text patterns like "Label: _____"
  - **Checkboxes**: Unicode checkbox symbols (☐) or `[ ]` patterns

**Backstory**:
```
You are a Template Analyzer agent specialized in analyzing template documents 
and extracting field definitions. When given a template, you:
1. Identify all fillable fields (empty cells, underscore lines, checkboxes)
2. Infer field meanings from context (section headers, nearby labels)
3. Create a structured TemplateSchema with field definitions
You return your analysis as a JSON TemplateSchema object.
```

### 2. Retriever Agent

**File**: `agents/retriever.py`

**Role**: Retrieves relevant context from indexed documents using RAG

**Responsibilities**:
- Execute RAG queries across document indexes
- Return retrieved context with citations
- Support document-specific or cross-document searches

**Tools**:
- `rag_retrieve_tool`: Custom CrewAI tool that wraps `rag_retrieve()` function

**Tool Function**:
```python
def rag_retrieve_tool_func(query: str, doc_id: str) -> str:
    """
    Retrieve context from RAG system.
    Returns JSON string with context, chunks, and citations.
    """
```

**Backstory**:
```
You are a Retriever agent specialized in retrieving relevant context from 
document indexes using RAG. When asked to retrieve information, you use the 
rag_retrieve tool to search indexed documents. You return retrieved context 
and citations in JSON format. You always include citations for all retrieved 
information.
```

**Implementation**:
- Uses `rag_adapter.rag_retrieve()` function
- Supports searching across all documents (doc_id=None) or specific documents
- Returns top 10 chunks with formatted context text
- Includes citation information for each chunk

### 3. Financial Extractor Agent

**File**: `agents/extract_financials.py`

**Role**: Extracts financial metrics and data from documents

**Responsibilities**:
- Extract financial values (amounts, ratios, percentages)
- Ensure precision with numbers
- Include citations for all extractions
- Return structured `FieldExtraction` objects

**Key Function**: `extract_financial_field(field_name, context, config) -> FieldExtraction`

**Prompt Used**: `get_financial_extraction_prompt()` from `prompts/field_prompts.py`

**Backstory**:
```
You are a Financial Extractor agent specialized in extracting financial metrics 
and data from documents. Your role is to extract financial values (amounts, ratios, 
percentages, etc.) from context. You return extractions as JSON FieldExtraction 
objects with value, confidence, and citations. You are precise with numbers and 
always include citations.
```

**Extraction Strategy**:
- Uses LLM directly (not CrewAI task execution)
- Prompts LLM with field name and context
- Parses JSON response to create `FieldExtraction` object
- Handles currency, numbers, percentages, ratios

### 4. Covenant Extractor Agent

**File**: `agents/extract_covenants.py`

**Role**: Extracts covenant terms and conditions from documents

**Responsibilities**:
- Extract covenant information from context
- Handle text-based fields
- Process checkbox fields (boolean values)
- Return structured extractions with citations

**Key Function**: `extract_covenant_field(field_name, context, config) -> FieldExtraction`

**Prompt Used**: `get_covenant_extraction_prompt()` from `prompts/field_prompts.py`

**Backstory**:
```
You are a Covenant Extractor agent specialized in extracting covenant terms 
and conditions from documents. Your role is to extract relevant covenant 
information from context. You return extractions as JSON FieldExtraction objects 
with value, confidence, and citations. You are thorough and always include citations.
```

**Extraction Strategy**:
- Similar to financial extractor but optimized for text/covenant fields
- Handles checkbox extraction (true/false values)
- Processes narrative and descriptive fields
- Default agent for most non-financial fields

### 5. Risk Synthesizer Agent

**File**: `agents/synthesize_risks.py`

**Role**: Synthesizes risk factors from multiple document sources

**Responsibilities**:
- Identify risks from multiple contexts
- Prioritize risks by severity
- Provide detailed risk descriptions
- Include citations for each risk

**Key Function**: `synthesize_risks(contexts, config) -> Dict`

**Prompt Used**: `get_risk_synthesis_prompt()` from `prompts/section_prompts.py`

**Backstory**:
```
You are a Risk Synthesizer agent specialized in synthesizing risk factors from 
multiple document sources. When given contexts from multiple documents, you 
identify and synthesize all significant risks. You return your synthesis as a 
JSON object with risks, severity levels, descriptions, and citations. You are 
comprehensive and prioritize risks by severity.
```

**Output Format**:
```json
{
  "risks": [
    {
      "risk_name": "Risk description",
      "severity": "low|medium|high",
      "description": "Detailed description",
      "citations": [...]
    }
  ]
}
```

### 6. Memo Writer Agent

**File**: `agents/memo_writer.py`

**Role**: Writes professional, well-structured narrative sections of credit memos

**Responsibilities**:
- Write narrative sections based on extracted fields
- Ensure professional tone and structure
- Include citations for factual claims
- Generate coherent, well-formatted text

**Key Function**: `write_section(section_name, extracted_fields, context, config) -> Dict`

**Prompt Used**: `get_section_writing_prompt()` from `prompts/section_prompts.py`

**Backstory**:
```
You are a Memo Writer agent specialized in writing narrative sections of credit 
memos. When given extracted fields and context, you write professional, 
well-structured memo sections. You return your writing as a JSON object with 
section content and citations. You are clear, professional, and always include 
citations for factual claims.
```

**Output Format**:
```json
{
  "section_name": "Section Name",
  "content": "Section text...",
  "citations": [...]
}
```

### 7. Consistency Checker Agent

**File**: `agents/consistency_checker.py`

**Role**: Validates consistency across extracted fields

**Responsibilities**:
- Check date alignment (dates are consistent)
- Verify number matching (numbers match across fields)
- Validate logical consistency (values make sense together)
- Perform cross-field validation

**Key Function**: `check_consistency(extracted_fields, config) -> List[ValidationRecord]`

**Backstory**:
```
You are a Consistency Checker agent specialized in validating consistency across 
extracted fields. You check for:
- Date alignment (dates are consistent)
- Number matching (numbers match across fields)
- Logical consistency (values make sense together)
- Cross-field validation
You return validation results as JSON ValidationRecord objects.
```

**Validation Checks**:
- Date consistency (e.g., loan date < maturity date)
- Number consistency (e.g., total = sum of parts)
- Logical relationships (e.g., ratios calculated correctly)
- Cross-reference validation

**Output**: List of `ValidationRecord` objects with status (pass/warning/error)

### 8. Policy Checker Agent

**File**: `agents/policy_checker.py`

**Role**: Validates extracted fields against credit policy rules

**Responsibilities**:
- Load policy rules from `policy_rules.json`
- Validate fields against policy thresholds
- Generate validation warnings/errors
- Return structured validation records

**Key Function**: `check_policy(extracted_fields, policy_rules_path, config) -> List[ValidationRecord]`

**Backstory**:
```
You are a Policy Checker agent specialized in validating extracted fields against 
credit policy rules. You check fields against policy rules (max/min values, ranges, 
required fields, etc.). You return validation results as JSON ValidationRecord objects.
```

**Policy Rules Format**:
```json
{
  "rules": [
    {
      "rule_id": "unique_id",
      "field": "field_name",
      "condition": "max|min|range",
      "value": 5.0,
      "severity": "error|warning",
      "message": "Human-readable message"
    }
  ]
}
```

**Validation Logic**:
- If `policy_rules.json` doesn't exist or is empty, returns empty list (no validation)
- Validates each rule against corresponding field
- Normalizes status values (handles edge cases like "pass|error")
- Returns list of `ValidationRecord` objects

### 9. Coordinator Agent

**File**: `agents/coordinator.py`

**Role**: Orchestrates the memo drafting workflow efficiently

**Responsibilities**:
- Create extraction plan (which fields, which agents)
- Assign tasks to specialist agents
- Collect structured results
- Trigger validations
- Assemble final memo draft
- Request human review

**Key Class**: `WorkflowState`

**Backstory**:
```
You are the Coordinator agent responsible for orchestrating the memo drafting workflow.
Your workflow:
1. Receive template schema from TemplateAnalyzer
2. Create extraction plan (which fields, which agents)
3. Assign tasks to specialist agents
4. Collect structured results
5. Trigger validations (ConsistencyChecker, PolicyChecker)
6. Assemble final memo draft
7. Request human review via HumanProxy
8. Run revision pass on feedback
You coordinate all other agents and manage the workflow state.
```

**WorkflowState Methods**:
- `create_extraction_plan() -> Dict[str, str]`: Maps field_id to agent_name
- `mark_task_complete(field_id)`: Updates task tracking
- `add_extraction(field_id, extraction)`: Stores extracted field
- `add_validation(record)`: Stores validation result
- `is_complete() -> bool`: Checks if all tasks done

**Extraction Plan Logic**:
- Financial fields (currency, number) → `financial_extractor`
- Covenant fields (text, checkbox) → `covenant_extractor`
- Default → `covenant_extractor`

### 10. Human Proxy Agent

**File**: `agents/human_proxy.py`

**Role**: Represents human-in-the-loop review

**Responsibilities**:
- Review memo drafts
- Provide feedback for revisions
- Ensure quality and accuracy
- Act as final authority

**Backstory**:
```
You are a Human Reviewer agent that represents human-in-the-loop review.
You review memo drafts and provide feedback for revisions.
You ensure quality and accuracy of the final memo.
```

**Note**: Currently a placeholder agent for future human-in-the-loop integration

---

## Prompts

All prompts are stored in the `prompts/` directory and are used by agents to guide LLM responses.

### Field Extraction Prompts

**File**: `prompts/field_prompts.py`

#### Financial Extraction Prompt

**Function**: `get_financial_extraction_prompt(field_name: str, context: str) -> str`

**Purpose**: Extract financial values (amounts, ratios, percentages) from context

**Prompt Structure**:
```
Extract the value for the field "{field_name}" from the following context.

Context:
{context}

Return a JSON object with the following structure:
{
    "field_id": "{field_name.lower().replace(' ', '_')}",
    "value": <extracted_value>,
    "confidence": <0.0-1.0>,
    "citations": [
        {
            "doc_id": "<document_id>",
            "page_span": [<start_page>, <end_page>],
            "section_label": "<section_name>",
            "citation_text": "<formatted_citation>"
        }
    ],
    "extraction_method": "financial_extractor"
}

If the value is a number or currency, ensure it's a number (not a string).
If the value cannot be found, set "value" to null and "confidence" to 0.0.
Always include citations for the source of the extracted value.
```

**Key Instructions**:
- Numbers must be numeric (not strings)
- Always include citations
- Set confidence to 0.0 if value not found

#### Covenant Extraction Prompt

**Function**: `get_covenant_extraction_prompt(field_name: str, context: str) -> str`

**Purpose**: Extract covenant terms and text-based information

**Similar structure to financial prompt but optimized for text/covenant fields**

#### Checkbox Extraction Prompt

**Function**: `get_checkbox_extraction_prompt(field_name: str, context: str) -> str`

**Purpose**: Determine if checkbox should be checked (true/false)

**Key Difference**: Returns boolean value instead of text/number

### Section Writing Prompts

**File**: `prompts/section_prompts.py`

#### Section Writing Prompt

**Function**: `get_section_writing_prompt(section_name, extracted_fields, context) -> str`

**Purpose**: Write narrative sections of credit memos

**Prompt Structure**:
```
Write the "{section_name}" section of a credit memo based on the following 
extracted fields and context.

Extracted Fields:
{fields_text}

Context:
{context}

Return a JSON object with the following structure:
{
    "section_name": "{section_name}",
    "content": "<section_text>",
    "citations": [...]
}

The section should be well-written, professional, and include citations for 
all factual claims.
```

#### Risk Synthesis Prompt

**Function**: `get_risk_synthesis_prompt(contexts: List[str]) -> str`

**Purpose**: Synthesize risks from multiple document contexts

**Prompt Structure**:
```
Synthesize risk factors from the following contexts:

{context_text}

Return a JSON object with the following structure:
{
    "risks": [
        {
            "risk_name": "<risk_description>",
            "severity": "<low|medium|high>",
            "description": "<detailed_description>",
            "citations": [...]
        }
    ]
}

Identify all significant risks and provide citations for each.
```

---

## Tools and Integrations

### RAG Retrieval Tool

**Location**: `agents/retriever.py`

**Tool Class**: `RAGRetrieveTool` (CrewAI BaseTool)

**Description**: "Retrieve relevant context from indexed documents using RAG. Takes a query string and document ID, returns JSON with context, chunks, and citations."

**Function**: `rag_retrieve_tool_func(query: str, doc_id: str) -> str`

**Implementation**:
1. Calls `rag_adapter.rag_retrieve()` with query and doc_id
2. Formats top 10 chunks as context text
3. Returns JSON with:
   - Query
   - Chunks (top 10)
   - Citations
   - Context pack (formatted text)

**Error Handling**: Returns error JSON if retrieval fails

### LLM Direct Call

**Location**: `agents/base.py`

**Function**: `call_agent_llm_direct(config, prompt, temperature, max_tokens) -> str`

**Purpose**: Call LLM directly (bypassing CrewAI task execution) for structured extraction

**Implementation**:
- Creates `OpenAILLMProvider` from config
- Calls `llm_provider.generate()` with prompt
- Returns raw text response

**Used By**:
- Financial Extractor
- Covenant Extractor
- Risk Synthesizer
- Memo Writer
- Consistency Checker
- Policy Checker

### ParseForge Integration

**Location**: `rag/rag_adapter.py`

**Functions**:
- `parse_to_markdown()`: Parse document to markdown
- `chunk_markdown()`: Chunk markdown into children/parents
- `index_chunks()`: Index chunks to vector store
- `rag_retrieve()`: Retrieve context using RAG

**Strategy Support**: All parsing functions support `StrategyEnum` (AUTO, LLM_FULL, etc.)

---

## Workflow

### Complete Workflow Steps

The `MemoIQOrchestrator.run()` method executes the following workflow:

#### Step 1: Template Analysis (10% progress)
- **Agent**: Template Analyzer
- **Action**: Parse template and detect fillable fields
- **Output**: `TemplateSchema` with field definitions
- **Saved**: `template_schema.json`

#### Step 2: Document Ingestion (20% progress)
- **Function**: `ingest_reference_documents()`
- **Actions**:
  1. Parse each document to markdown
  2. Chunk markdown into children/parents
  3. Index chunks to vector store
- **Output**: Dict mapping doc_id → (children, parents, stats)
- **Strategy**: Uses `config.parsing_strategy` (default: AUTO)

#### Step 3: Agent Initialization (40% progress)
- **Agents Created**:
  - Coordinator
  - Retriever (with all_chunks for neighbor expansion)
  - Financial Extractor
  - Covenant Extractor
  - Risk Synthesizer
  - Memo Writer
  - Consistency Checker
  - Policy Checker
  - Human Proxy

#### Step 4: Extraction Plan Creation (50% progress)
- **Action**: `WorkflowState.create_extraction_plan()`
- **Logic**: Maps each field_id to appropriate agent
  - Financial fields → `financial_extractor`
  - Covenant/text fields → `covenant_extractor`
  - Checkboxes → `covenant_extractor`

#### Step 5: Field Extraction (60% progress)
- **For each field**:
  1. Build query: `"{field_name}: {description}"`
  2. Retrieve context: `rag_retrieve(query, doc_id=None)` (searches all docs)
  3. Extract field value using appropriate agent
  4. Store extraction in `WorkflowState`
- **Error Handling**:
  - Rate limit errors: Logged, empty extraction added
  - Other errors: Logged, empty extraction added
  - Workflow continues with remaining fields

#### Step 6: Validation (80% progress)
- **Consistency Check**:
  - Calls `check_consistency(extracted_fields, config)`
  - Adds validation records to workflow state
- **Policy Check** (if `policy_rules.json` exists):
  - Calls `check_policy(extracted_fields, policy_rules_path, config)`
  - Adds validation records to workflow state
- **Output**: List of `ValidationRecord` objects

#### Step 7: Template Filling (90% progress)
- **Class**: `TemplateFiller`
- **Actions**:
  1. Load template (DOCX or convert PDF to DOCX)
  2. Fill each field in its original location
  3. Preserve structure (tables, paragraphs, formatting)
  4. Add citations to filled fields
- **Output**: `draft_v1.docx`

#### Step 8: Evidence Pack Creation
- **Action**: Create `EvidencePack` for each field
- **Contains**: Value, citations, extraction context, confidence

#### Step 9: Memo Draft Assembly
- **Creates**: `MemoDraft` object with:
  - Draft DOCX path
  - Extracted fields
  - Validation report
  - Evidence pack
  - Metadata (version, timestamps)

#### Step 10: Save Results
- **Files Saved**:
  - `extracted_fields.json`
  - `validation_report.json`
  - `evidence_pack.json`
  - `outputs/draft_v1.docx`

### Error Handling

**Rate Limit Errors**:
- Detected by checking error message for "429", "rate limit", or "quota"
- Logged with warning
- Empty extraction added to continue workflow
- Small delay (0.5s) added to avoid hammering API

**Other Errors**:
- Logged with full traceback
- Empty extraction added to continue workflow
- Summary logged at end of extraction phase

**Error Summary**:
- Logs count of rate limit errors vs other errors
- Lists affected fields (up to 10)
- Provides actionable guidance

---

## Template System

### Template Parser

**File**: `template/template_parser.py`

**Class**: `TemplateParser`

#### DOCX Template Parsing

**Method**: `_parse_docx_template(template_path) -> TemplateSchema`

**Field Detection**:

1. **Empty Cells**:
   - Detects cells with no text or only whitespace
   - Infers field name from:
     - Row header (first cell in row)
     - Column header (first row)
     - Previous cell in same row
   - Creates `FieldDefinition` with location (table_id, row, col)

2. **Underscore Lines**:
   - Detects patterns like "Label: _____"
   - Extracts label as field name
   - Creates `FieldDefinition` with location (paragraph_idx, run_idx)

3. **Checkboxes**:
   - Detects Unicode checkbox (☐) or `[ ]` patterns
   - Extracts checkbox text as field name
   - Creates `FieldDefinition` with type="checkbox"

**Field Type Inference**:
- Currency: Contains "amount", "price", "cost", "fee", "loan"
- Date: Contains "date"
- Number: Contains "number", "count", "quantity"
- Text: Default

#### PDF Template Parsing

**Method**: `_parse_pdf_template(template_path) -> TemplateSchema`

**Process**:
1. Parse PDF to markdown using `parse_to_markdown()`
2. Detect tables in markdown (markdown table syntax)
3. Detect empty cells in tables
4. Detect underscore lines in text
5. Extract sections from markdown headings

**Table Detection**:
- Looks for markdown table rows: `| col1 | col2 | col3 |`
- Empty cells: `| col1 |  | col3 |` (empty between pipes)
- Uses header row for field name inference

**Section Extraction**:
- Detects markdown headings: `# Heading`, `## Subheading`, etc.
- Creates section structure with level

### Template Filler

**File**: `template/template_filler.py`

**Class**: `TemplateFiller`

#### DOCX Template Filling

**Method**: `_fill_docx_template(template_schema, extracted_fields, output_path) -> str`

**Process**:
1. Load original DOCX template
2. Fill table cells:
   - Locate cell by (table_id, row, col)
   - Clear cell content
   - Add extracted value
   - Add citation as footnote or inline
3. Fill paragraphs:
   - Locate paragraph by paragraph_idx
   - Replace underscore pattern with value
   - Add citation inline
4. Update checkboxes:
   - Replace ☐ with ☑ if value is True
5. Save filled document

**Structure Preservation**:
- Maintains all formatting (fonts, styles, alignment)
- Preserves table structure
- Keeps paragraph structure
- Preserves document sections

#### PDF Template Filling

**Method**: `_fill_pdf_template(template_schema, extracted_fields, output_path) -> str`

**Process**:
1. Parse PDF to markdown
2. Replace field placeholders with extracted values
3. Add citations to markdown
4. Convert markdown to DOCX (simplified conversion)
5. Save as DOCX

**Note**: PDF templates are always converted to DOCX for output

### Citation Formatting

**Method**: `_format_citations(citations) -> str`

**Format**:
- Uses `citation_text` if available
- Otherwise: `"Doc: {doc_id} | P{page_start}-{page_end}"`
- Limits to 3 citations per field
- Joins with "; " separator

---

## RAG Integration

### RAG Adapter

**File**: `rag/rag_adapter.py`

**Purpose**: Thin wrappers around `/src` RAG functions - NO logic duplication

#### Parse to Markdown

**Function**: `parse_to_markdown(file_path, config, generate_image_descriptions, strategy) -> str`

**Implementation**:
- Creates `ParseForge` instance
- Calls `parser.parse(file_path, strategy=strategy)`
- Returns `parser.to_markdown(doc, generate_image_descriptions=...)`

**Strategy Support**:
- `StrategyEnum.AUTO`: Default, uses layout detection
- `StrategyEnum.LLM_FULL`: Bypasses layout detection, uses LLM only
- Logs strategy being used for debugging

#### Chunk Markdown

**Function**: `chunk_markdown(md_path, doc_id, config) -> Tuple[List[Chunk], List[ParentChunk], Dict]`

**Implementation**:
- Calls `process_document(md_path, config, doc_id)` from chunking pipeline
- Returns (children, parents, stats)

#### Index Chunks

**Function**: `index_chunks(children, parents, doc_id, config) -> Dict`

**Process**:
1. Validate chunks (check for chunk_id, text_for_embedding)
2. Serialize to temporary JSONL files
3. Call `ingest_from_chunking_outputs(children_path, parents_path, doc_id, config)`
4. Clean up temp files
5. Return indexing stats

**Error Handling**:
- Logs warnings for invalid chunks
- Returns error dict if indexing fails

#### RAG Retrieve

**Function**: `rag_retrieve(query, doc_id, config, all_chunks, llm_config) -> ContextPack`

**Implementation**:
- If `doc_id` is None: Search across all documents (no filter)
- If `doc_id` provided: Filter to specific document
- Calls `retrieve(query, filters, config, all_chunks, llm_config=llm_config)`
- Returns `ContextPack` with selected chunks, citations, trace

**Neighbor Expansion**:
- Uses `all_chunks` parameter for neighbor expansion
- Expands context by including neighboring chunks

### Document Ingestion

**File**: `rag/doc_ingest.py`

**Function**: `ingest_reference_documents(document_paths, parsing_config, chunking_config, retrieval_config, index_documents, strategy) -> Dict`

**Process** (per document):
1. **Parse**: `parse_to_markdown(doc_path, parsing_config, strategy=strategy)`
2. **Save to temp file**: Write markdown to temporary file
3. **Chunk**: `chunk_markdown(tmp_md_path, doc_id, chunking_config)`
4. **Index** (if requested): `index_chunks(children, parents, doc_id, retrieval_config)`
5. **Clean up**: Delete temp markdown file

**Returns**: Dict mapping `doc_id → (children, parents, stats)`

**Error Handling**:
- Continues with other documents if one fails
- Returns empty chunks and error in stats

---

## Validation System

### Consistency Checker

**File**: `agents/consistency_checker.py`

**Function**: `check_consistency(extracted_fields, config) -> List[ValidationRecord]`

**Process**:
1. Serialize extracted fields to JSON (value and type only)
2. Prompt LLM to check consistency
3. Parse JSON array of `ValidationRecord` objects
4. Return list of validation records

**Prompt Structure**:
```
Check consistency of the following extracted fields:

{fields_json}

Return a JSON array of ValidationRecord objects:
[
    {
        "field_id": "<field_id_or_null_for_global>",
        "status": "pass|warning|error",
        "message": "<validation_message>",
        "suggestions": ["<suggestion1>", "<suggestion2>"],
        "severity": <0-10>
    }
]
```

**Validation Checks**:
- Date alignment
- Number matching
- Logical consistency
- Cross-field validation

### Policy Checker

**File**: `agents/policy_checker.py`

**Function**: `check_policy(extracted_fields, policy_rules_path, config) -> List[ValidationRecord]`

**Process**:
1. Load `policy_rules.json` (if exists)
2. If no rules or empty: Return empty list (skip validation)
3. Serialize fields and rules to JSON
4. Prompt LLM to validate fields against rules
5. Parse and normalize validation records
6. Return list of validation records

**Policy Rules Format**:
```json
{
  "rules": [
    {
      "rule_id": "unique_id",
      "field": "field_name",
      "condition": "max|min|range",
      "value": 5.0,
      "min_value": 0.0,
      "max_value": 1.0,
      "severity": "error|warning",
      "message": "Human-readable message"
    }
  ]
}
```

**Status Normalization**:
- Handles edge cases like "pass|error" → takes most severe
- Ensures status is one of: "pass", "warning", "error"
- Defaults to "warning" if invalid

**Prompt Structure**:
```
Check the following extracted fields against policy rules:

Fields:
{fields_json}

Policy Rules:
{rules_json}

Return a JSON array of ValidationRecord objects...
```

### Validation Record

**Schema**: `ValidationRecord` (from `schema.py`)

**Fields**:
- `field_id`: Optional[str] - Field ID or None for global validations
- `status`: Literal["pass", "warning", "error"]
- `message`: str - Validation message
- `suggestions`: List[str] - Suggested fixes
- `severity`: int (0-10) - 0=info, 5=warning, 10=error

---

## Configuration

### MemoIQConfig

**File**: `config.py`

**Class**: `MemoIQConfig`

**Fields**:
- `runs_dir`: Path - Directory for storing run artifacts (default: `MemoIQ/runs`)
- `llm_provider`: Optional[str] - LLM provider override (None uses .env defaults)
- `agent_llm_config`: Optional[ParseForgeConfig] - LLM config for agents
- `parsing_config`: Optional[ParseForgeConfig] - RAG parsing config
- `chunking_config`: Optional[ChunkConfig] - RAG chunking config
- `retrieval_config`: Optional[RetrievalConfig] - RAG retrieval config
- `parsing_strategy`: Optional[StrategyEnum] - Parsing strategy
- `agent_temperature`: float - LLM temperature for agents (default: 0.7)
- `agent_max_turns`: int - Maximum conversation turns (default: 50)
- `agent_timeout`: int - Agent timeout in seconds (default: 300)

**Initialization**:
- Creates configs from environment if not provided
- Handles LLM provider override
- Ensures runs directory exists

### Agent Configuration

**File**: `agents/base.py`

**Function**: `create_agent(name, role, goal, backstory, config, tools, verbose, allow_delegation) -> Agent`

**LLM Setup**:
- Creates `OpenAILLMProvider` from `config.agent_llm_config`
- Sets environment variables for CrewAI:
  - Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_KEY`
  - OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`
- Sets `OPENAI_TEMPERATURE` from config
- Creates LangChain LLM if available (AzureChatOpenAI or ChatOpenAI)
- Falls back to environment variables if LangChain not available

**Agent Creation**:
- Uses CrewAI `Agent` class
- Sets role, goal, backstory
- Adds tools if provided
- Configures delegation (coordinator can delegate)

---

## Data Models

### Schema Definitions

**File**: `schema.py`

#### FieldDefinition

**Purpose**: Metadata for a fillable field in a template

**Fields**:
- `field_id`: str - Unique field identifier
- `name`: str - Semantic field name (e.g., "Applicant Name", "Loan Amount")
- `type`: Literal["text", "number", "date", "boolean", "checkbox", "currency", "list", "table"]
- `required`: bool - Whether field is required (default: True)
- `description`: Optional[str] - Field description
- `location`: Dict[str, Any] - Position in template (table_id, row, col) or (paragraph_idx, run_idx)
- `field_type`: Literal["empty_cell", "underscore_line", "checkbox", "narrative"] - Detection method
- `context`: Optional[str] - Nearby labels/headers used to infer field meaning
- `placeholder_text`: Optional[str] - Original placeholder or empty cell indicator

#### TemplateSchema

**Purpose**: Complete parsed template structure

**Fields**:
- `template_path`: str - Path to template file
- `template_format`: Literal["pdf", "docx"] - Template format
- `sections`: List[Dict[str, Any]] - List of sections with detected fillable fields
- `field_definitions`: Dict[str, FieldDefinition] - field_id → FieldDefinition
- `structure_map`: Dict[str, Any] - Document structure metadata

#### Citation

**Purpose**: Citation information for extracted values

**Fields**:
- `doc_id`: str - Document ID
- `chunk_id`: Optional[str] - Chunk ID
- `page_span`: Tuple[int, int] - Start and end page numbers
- `section_label`: Optional[str] - Section name
- `header_path`: Optional[str] - Header path
- `citation_text`: Optional[str] - Formatted citation string

#### FieldExtraction

**Purpose**: Extracted field value with metadata

**Fields**:
- `field_id`: str - Field identifier
- `value`: Union[str, int, float, bool, List[Any], Dict[str, Any], None] - Extracted value
- `confidence`: float (0.0-1.0) - Extraction confidence
- `citations`: List[Citation] - Source citations
- `extraction_method`: Optional[str] - Which agent extracted this
- `raw_context`: Optional[str] - Raw context used for extraction
- `field_type`: Optional[str] - Field type (empty_cell, underscore_line, checkbox)

#### ValidationRecord

**Purpose**: Validation result

**Fields**:
- `field_id`: Optional[str] - Field ID or None for global validations
- `status`: Literal["pass", "warning", "error"] - Validation status
- `message`: str - Validation message
- `suggestions`: List[str] - Suggested fixes
- `severity`: int (0-10) - 0=info, 5=warning, 10=error

#### EvidencePack

**Purpose**: Evidence pack for a field

**Fields**:
- `field_id`: str - Field identifier
- `value`: Union[str, int, float, bool, List[Any], Dict[str, Any], None] - Field value
- `citations`: List[Citation] - Source citations
- `extraction_context`: Optional[str] - Context used for extraction
- `confidence`: float (0.0-1.0) - Extraction confidence

#### MemoDraft

**Purpose**: Complete memo draft structure

**Fields**:
- `draft_docx_path`: str - Path to filled DOCX file (always DOCX format)
- `extracted_fields`: Dict[str, FieldExtraction] - field_id → FieldExtraction
- `validation_report`: List[ValidationRecord] - Validation results
- `evidence_pack`: Dict[str, EvidencePack] - field_id → EvidencePack
- `structure_preserved`: bool - Confirms structure matches template (default: True)
- `draft_version`: int - Draft version number (default: 1)
- `created_at`: Optional[str] - Creation timestamp
- `updated_at`: Optional[str] - Update timestamp

---

## Utilities

### I/O Utilities

**File**: `utils/io.py`

**Functions**:
- `create_run_directory(config) -> Path`: Create new run directory with subdirectories
- `save_template(run_dir, template_path) -> Path`: Save template to run directory
- `save_reference_docs(run_dir, doc_paths) -> List[Path]`: Save reference documents
- `save_json(run_dir, filename, data) -> Path`: Save JSON data
- `load_json(run_dir, filename) -> Any`: Load JSON data
- `save_draft(run_dir, draft_path, version) -> Path`: Save draft DOCX

### Citation Utilities

**File**: `utils/citations.py`

**Functions**:
- `format_citation_from_chunk(chunk) -> str`: Format citation from chunk dictionary
- `format_citation_from_evidence(evidence) -> str`: Format citation from evidence pack

**Implementation**: Wraps `src.utils.io.format_citation()`

### Diff Utilities

**File**: `utils/diff.py`

**Functions**:
- `compare_drafts(draft_v1, draft_v2) -> Dict`: Compare two drafts and generate delta log
- `save_delta_log(run_dir, delta, version) -> Path`: Save delta log to run directory

**Delta Log Structure**:
```json
{
  "timestamp": "ISO timestamp",
  "from_version": 1,
  "to_version": 2,
  "changed_fields": [...],
  "added_fields": [...],
  "removed_fields": [...],
  "validation_changes": [...]
}
```

### Metrics Utilities

**File**: `utils/metrics.py`

**Functions**:
- `calculate_extraction_confidence(draft) -> float`: Calculate average extraction confidence (0.0-1.0)
- `calculate_validation_score(draft) -> Dict`: Calculate validation score with pass/warning/error counts
- `calculate_completeness(draft, template_schema) -> float`: Calculate field completeness (0.0-1.0)

**Validation Score**:
- Pass = 1.0
- Warning = 0.5
- Error = 0.0
- Score = (pass_count * 1.0 + warning_count * 0.5) / total

---

## Run Management

### Run Directory Structure

Each MemoIQ run creates a directory under `MemoIQ/runs/{run_id}/` with:

```
{run_id}/
├── template/
│   └── {template_filename}          # Template file
├── reference_docs/
│   ├── {doc1_filename}               # Reference documents
│   └── {doc2_filename}
├── outputs/
│   ├── draft_v1.docx                 # First filled draft
│   └── delta_log_v2.json             # Changes between versions (if revised)
├── template_schema.json              # Parsed template schema
├── extracted_fields.json            # All extracted fields
├── validation_report.json            # Validation results
└── evidence_pack.json                # Field → citations mapping
```

### Run Lifecycle

1. **Creation**: `create_run_directory()` creates new run with UUID
2. **Template Analysis**: Template saved and schema generated
3. **Document Ingestion**: Reference documents saved and indexed
4. **Field Extraction**: Extractions stored in `extracted_fields.json`
5. **Validation**: Validation results stored in `validation_report.json`
6. **Template Filling**: Draft saved to `outputs/draft_v1.docx`
7. **Evidence Pack**: Evidence stored in `evidence_pack.json`

### Run Artifacts

**template_schema.json**:
- Complete template structure
- All field definitions
- Document structure map

**extracted_fields.json**:
- Dict mapping field_id → FieldExtraction
- Includes value, confidence, citations, extraction_method

**validation_report.json**:
- List of ValidationRecord objects
- Includes consistency and policy validations

**evidence_pack.json**:
- Dict mapping field_id → EvidencePack
- Includes value, citations, extraction_context, confidence

**draft_v1.docx**:
- Filled template with all extracted values
- Citations included inline
- Structure preserved from original template

---

## Usage Examples

### Basic Usage

```python
from MemoIQ.agents.orchestrator import MemoIQOrchestrator
from MemoIQ.config import MemoIQConfig

# Create configuration
config = MemoIQConfig()

# Create orchestrator
orchestrator = MemoIQOrchestrator(config)

# Run workflow
results = orchestrator.run(
    template_path="template.pdf",
    reference_doc_paths=["doc1.pdf", "doc2.docx"],
)

# Access results
draft = results["memo_draft"]
print(f"Draft saved to: {draft.draft_docx_path}")
print(f"Extracted {len(draft.extracted_fields)} fields")
print(f"Validation: {len(draft.validation_report)} records")
```

### With Progress Callback

```python
def progress_callback(stage: str, progress: float):
    print(f"{stage}: {progress*100:.1f}%")

results = orchestrator.run(
    template_path="template.pdf",
    reference_doc_paths=["doc1.pdf"],
    progress_callback=progress_callback,
)
```

### Custom Configuration

```python
from MemoIQ.config import MemoIQConfig
from src.config.parsing_strategies import StrategyEnum

config = MemoIQConfig(
    runs_dir=Path("custom/runs"),
    llm_provider="azure_openai",
    parsing_strategy=StrategyEnum.LLM_FULL,
    agent_temperature=0.5,
    agent_max_turns=100,
)

orchestrator = MemoIQOrchestrator(config)
```

---

## Best Practices

### Template Design

1. **Use Clear Labels**: Field names are inferred from nearby labels - use clear, descriptive labels
2. **Consistent Structure**: Maintain consistent table structures for better field detection
3. **Avoid Ambiguity**: Use distinct labels to avoid field name conflicts
4. **Checkbox Format**: Use Unicode checkbox (☐) or `[ ]` for checkbox fields

### Document Preparation

1. **High Quality**: Use high-quality PDFs/DOCX files for better parsing
2. **Structured Content**: Well-structured documents improve extraction accuracy
3. **Complete Information**: Ensure reference documents contain all required information
4. **Multiple Sources**: Provide multiple reference documents for better coverage

### Configuration

1. **LLM Provider**: Choose appropriate provider (OpenAI vs Azure OpenAI)
2. **Parsing Strategy**: Use AUTO for most cases, LLM_FULL for complex documents
3. **Temperature**: Lower temperature (0.3-0.5) for more consistent extractions
4. **Timeout**: Increase timeout for large documents or slow networks

### Validation

1. **Policy Rules**: Create comprehensive policy rules for your domain
2. **Review Validation Reports**: Always review validation reports before finalizing
3. **Consistency Checks**: Ensure consistency checks catch logical errors
4. **Human Review**: Always perform human review before finalizing memos

---

## Troubleshooting

### Common Issues

**Rate Limit Errors**:
- **Symptom**: "Rate limit error" in logs
- **Cause**: OpenAI API quota exceeded
- **Solution**: Check billing, increase quota, or add delays between requests

**Empty Extractions**:
- **Symptom**: Fields extracted as null
- **Cause**: Information not found in reference documents
- **Solution**: Review reference documents, improve query construction

**Template Parsing Errors**:
- **Symptom**: Fields not detected correctly
- **Cause**: Template structure not recognized
- **Solution**: Use clearer labels, check template format

**Validation Failures**:
- **Symptom**: Many validation errors
- **Cause**: Extracted values don't match policy rules
- **Solution**: Review policy rules, check extracted values, improve extraction prompts

### Debugging

**Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Run Artifacts**:
- Review `extracted_fields.json` for extraction results
- Review `validation_report.json` for validation issues
- Review `evidence_pack.json` for citation sources

**Test Individual Components**:
- Test template parsing separately
- Test RAG retrieval separately
- Test field extraction separately

---

## Future Enhancements

### Planned Features

1. **Human-in-the-Loop**: Full integration of human review and revision workflow
2. **Multi-Version Support**: Better support for draft revisions and versioning
3. **Custom Prompts**: Allow users to customize extraction prompts
4. **Batch Processing**: Support for processing multiple templates at once
5. **Advanced Validation**: More sophisticated validation rules and checks
6. **Export Formats**: Support for additional output formats (PDF, HTML)

### Extension Points

1. **Custom Agents**: Add custom agents for domain-specific extraction
2. **Custom Validators**: Add custom validation logic
3. **Custom Templates**: Support for additional template formats
4. **Custom Storage**: Support for additional vector stores

---

## Conclusion

MemoIQ is a comprehensive system for automating credit memo drafting. It combines multi-agent orchestration, RAG-based retrieval, intelligent template parsing, and robust validation to create a powerful workflow automation tool.

For questions or issues, refer to:
- Main README: `MemoIQ/README.md`
- Policy Rules: `MemoIQ/policy/README.md`
- Source code documentation in individual files

