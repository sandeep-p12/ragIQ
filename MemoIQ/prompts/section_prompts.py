"""Section writing prompts."""

from typing import Dict, List


def get_section_writing_prompt(
    section_name: str,
    extracted_fields: Dict[str, any],
    context: str,
) -> str:
    """Get prompt for writing a memo section."""
    fields_text = "\n".join([f"- {k}: {v}" for k, v in extracted_fields.items()])
    
    return f"""Write the "{section_name}" section of a credit memo based on the following extracted fields and context.

Extracted Fields:
{fields_text}

Context:
{context}

Return a JSON object with the following structure:
{{
    "section_name": "{section_name}",
    "content": "<section_text>",
    "citations": [
        {{
            "doc_id": "<document_id>",
            "page_span": [<start_page>, <end_page>],
            "section_label": "<section_name>",
            "citation_text": "<formatted_citation>"
        }}
    ]
}}

The section should be well-written, professional, and include citations for all factual claims.
"""


def get_risk_synthesis_prompt(contexts: List[str]) -> str:
    """Get prompt for synthesizing risks."""
    context_text = "\n\n---\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
    
    return f"""Synthesize risk factors from the following contexts:

{context_text}

Return a JSON object with the following structure:
{{
    "risks": [
        {{
            "risk_name": "<risk_description>",
            "severity": "<low|medium|high>",
            "description": "<detailed_description>",
            "citations": [
                {{
                    "doc_id": "<document_id>",
                    "page_span": [<start_page>, <end_page>],
                    "section_label": "<section_name>",
                    "citation_text": "<formatted_citation>"
                }}
            ]
        }}
    ]
}}

Identify all significant risks and provide citations for each.
"""

