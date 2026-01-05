"""Pydantic models for MemoIQ."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class FieldDefinition(BaseModel):
    """Field metadata definition."""
    
    field_id: str
    name: str  # Semantic field name (e.g., "Applicant Name", "Loan Amount")
    type: Literal["text", "number", "date", "boolean", "checkbox", "currency", "list", "table"] = "text"
    required: bool = True
    description: Optional[str] = None
    location: Dict[str, Any]  # Position in template (table_id, row, col) or (paragraph_idx, run_idx)
    field_type: Literal["empty_cell", "underscore_line", "checkbox", "narrative"]  # Detection method
    context: Optional[str] = None  # Nearby labels/headers used to infer field meaning
    placeholder_text: Optional[str] = None  # Original placeholder or empty cell indicator


class TemplateSchema(BaseModel):
    """Parsed template structure."""
    
    template_path: str
    template_format: Literal["pdf", "docx"]
    sections: List[Dict[str, Any]]  # List of sections with detected fillable fields
    field_definitions: Dict[str, FieldDefinition]  # field_id -> FieldDefinition
    structure_map: Dict[str, Any]  # Document structure metadata
    # Field locations: Maps field_id -> location (table_id, row, col) or (paragraph_idx, run_idx)
    # Field types: Detected from context
    # Formatting metadata: Styles, fonts, alignment, borders


class Citation(BaseModel):
    """Citation information."""
    
    doc_id: str
    chunk_id: Optional[str] = None
    page_span: Tuple[int, int]
    section_label: Optional[str] = None
    header_path: Optional[str] = None
    citation_text: Optional[str] = None  # Formatted citation string


class FieldExtraction(BaseModel):
    """Extracted field value with citations."""
    
    field_id: str
    value: Union[str, int, float, bool, List[Any], Dict[str, Any], None]
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    citations: List[Citation] = Field(default_factory=list)
    extraction_method: Optional[str] = None  # Which agent extracted this
    raw_context: Optional[str] = None  # Raw context used for extraction
    field_type: Optional[str] = None  # Field type (empty_cell, underscore_line, checkbox)


class ValidationRecord(BaseModel):
    """Validation result."""
    
    field_id: Optional[str] = None  # None for global validations
    status: Literal["pass", "warning", "error"]
    message: str
    suggestions: List[str] = Field(default_factory=list)
    severity: int = Field(ge=0, le=10, default=5)  # 0=info, 5=warning, 10=error


class EvidencePack(BaseModel):
    """Evidence pack for a field."""
    
    field_id: str
    value: Union[str, int, float, bool, List[Any], Dict[str, Any], None]
    citations: List[Citation]
    extraction_context: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class MemoDraft(BaseModel):
    """Complete memo draft structure."""
    
    draft_docx_path: str  # Path to filled DOCX file (always DOCX format)
    extracted_fields: Dict[str, FieldExtraction]  # field_id -> FieldExtraction
    validation_report: List[ValidationRecord]
    evidence_pack: Dict[str, EvidencePack]  # field_id -> EvidencePack
    structure_preserved: bool = True  # Confirms structure matches template
    draft_version: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

