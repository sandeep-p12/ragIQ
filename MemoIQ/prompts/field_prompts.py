"""Field extraction prompts with strict JSON schema."""

from typing import Dict


def get_financial_extraction_prompt(field_name: str, context: str) -> str:
    """Get prompt for financial field extraction."""
    return f"""Extract the value for the field "{field_name}" from the following context.

Context:
{context}

Return a JSON object with the following structure:
{{
    "field_id": "{field_name.lower().replace(' ', '_')}",
    "value": <extracted_value>,
    "confidence": <0.0-1.0>,
    "citations": [
        {{
            "doc_id": "<document_id>",
            "page_span": [<start_page>, <end_page>],
            "section_label": "<section_name>",
            "citation_text": "<formatted_citation>"
        }}
    ],
    "extraction_method": "financial_extractor"
}}

If the value is a number or currency, ensure it's a number (not a string).
If the value cannot be found, set "value" to null and "confidence" to 0.0.
Always include citations for the source of the extracted value.
"""


def get_covenant_extraction_prompt(field_name: str, context: str) -> str:
    """Get prompt for covenant field extraction."""
    return f"""Extract covenant information for the field "{field_name}" from the following context.

Context:
{context}

Return a JSON object with the following structure:
{{
    "field_id": "{field_name.lower().replace(' ', '_')}",
    "value": <extracted_value_or_description>,
    "confidence": <0.0-1.0>,
    "citations": [
        {{
            "doc_id": "<document_id>",
            "page_span": [<start_page>, <end_page>],
            "section_label": "<section_name>",
            "citation_text": "<formatted_citation>"
        }}
    ],
    "extraction_method": "covenant_extractor"
}}

If the value cannot be found, set "value" to null and "confidence" to 0.0.
Always include citations for the source of the extracted value.
"""


def get_text_extraction_prompt(field_name: str, context: str) -> str:
    """Get prompt for text field extraction."""
    return f"""Extract the value for the text field "{field_name}" from the following context.

Context:
{context}

Return a JSON object with the following structure:
{{
    "field_id": "{field_name.lower().replace(' ', '_')}",
    "value": "<extracted_text>",
    "confidence": <0.0-1.0>,
    "citations": [
        {{
            "doc_id": "<document_id>",
            "page_span": [<start_page>, <end_page>],
            "section_label": "<section_name>",
            "citation_text": "<formatted_citation>"
        }}
    ],
    "extraction_method": "text_extractor"
}}

If the value cannot be found, set "value" to null and "confidence" to 0.0.
Always include citations for the source of the extracted value.
"""


def get_checkbox_extraction_prompt(field_name: str, context: str) -> str:
    """Get prompt for checkbox field extraction."""
    return f"""Determine if the checkbox field "{field_name}" should be checked based on the following context.

Context:
{context}

Return a JSON object with the following structure:
{{
    "field_id": "{field_name.lower().replace(' ', '_')}",
    "value": <true_or_false>,
    "confidence": <0.0-1.0>,
    "citations": [
        {{
            "doc_id": "<document_id>",
            "page_span": [<start_page>, <end_page>],
            "section_label": "<section_name>",
            "citation_text": "<formatted_citation>"
        }}
    ],
    "extraction_method": "checkbox_extractor"
}}

If the value cannot be determined, set "value" to false and "confidence" to 0.0.
Always include citations for the source of the determination.
"""

