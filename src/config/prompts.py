"""Centralized prompts for LLM operations in ParseForge.

This module contains all prompts used throughout the codebase for:
- PDF parsing and transcription
- Image description generation
- Table extraction and formatting
- Page processing with OCR
"""

from typing import Optional

# ============================================================================
# PDF Parsing Prompts
# ============================================================================

BASE_LLM_PROMPT = """You are tasked with transcribing and formatting the content of a file into markdown. Your goal is to create a well-structured, readable markdown document that accurately represents the original content while adding appropriate formatting and identifier tags.

CRITICAL: You MUST include identifier tags for tables, images, headers, and table of contents. These tags are essential for proper document structure.

Follow these instructions to complete the task:

1. Carefully read through the entire file content.

2. Transcribe the content into markdown format, paying close attention to the existing formatting and structure.

3. If you encounter any unclear formatting in the original content, use your judgment to add appropriate markdown formatting to improve readability and structure.

4. IDENTIFIER TAGS - You MUST use these tags:
   
   a) TABLES:
   - For ALL tables (simple, complex, multi-page): Enclose the entire table in [TABLE] and [/TABLE] tags
   - Merge content of tables if it continues across multiple pages into a single table
   - Use proper markdown table formatting with pipes (|) and hyphens (-) for table structure
   - Example:
     [TABLE]
     | Column 1 | Column 2 | Column 3 |
     |----------|----------|----------|
     | Data 1   | Data 2   | Data 3   |
     [/TABLE]
   
   b) IMAGES, CHARTS, DIAGRAMS:
   - For ALL images, charts, graphs, diagrams, flowcharts, KPIs, and visualizations: Enclose descriptions in [IMAGE] and [/IMAGE] tags
   - Provide detailed descriptions including:
     * For charts/graphs: Data values, trends, axes labels, legends, units, insights
     * For diagrams: Structure, connections, flow, elements, relationships
     * For KPIs/metrics: Values, labels, context, comparisons
     * For flowcharts: Steps, decision points, flow direction
   - Place image descriptions at the exact position where the image appears in the document
   - Example:
     [IMAGE]
     Bar chart showing quarterly sales: Q1 ($50K), Q2 ($65K), Q3 ($72K), Q4 ($80K). 
     X-axis: Quarters, Y-axis: Sales in thousands. Trend: Steady upward growth.
     [/IMAGE]
   
   c) HEADERS:
   - For headers/footers (complete chain of characters repeated at the start/end of each page): Enclose in [HEADER] and [/HEADER] tags
   - Only tag repeated headers/footers, not section headers
   
   d) TABLE OF CONTENTS:
   - If a table of contents is present: Enclose it in [TOC] and [/TOC] tags
   - Format as a proper markdown list with page numbers

5. TABLE HANDLING:
   - Simple tables: Use standard markdown table format with proper alignment
   - Complex tables: Preserve structure, use markdown table format, ensure all rows have the same number of columns
   - Multi-page tables: Merge into single table, maintain column alignment across pages
   - Tables with merged cells: Represent clearly in markdown (repeat content or use appropriate formatting)
   - CRITICAL: All rows in a table MUST have the same number of columns
   - Always include a separator row (| --- | --- |) after the header row
   - Always wrap in [TABLE]...[/TABLE] tags
   - Example of properly formatted table:
     [TABLE]
     | Column 1 | Column 2 | Column 3 |
     |----------|----------|----------|
     | Data 1   | Data 2   | Data 3   |
     | Data 4   | Data 5   | Data 6   |
     [/TABLE]

6. IMAGE HANDLING:
   - Identify ALL images, charts, diagrams, graphs, flowcharts, KPIs
   - Provide comprehensive descriptions that capture all information
   - Include quantitative data when present (numbers, percentages, values)
   - Describe visual relationships, trends, and patterns
   - Always wrap descriptions in [IMAGE]...[/IMAGE] tags

7. PAGE IDENTIFIERS - CRITICAL:
   - You MUST add page identifiers to clearly separate content from different pages
   - At the START of content from each page, add a page identifier in this exact format:
     --- Page X ---
   - Use individual page identifiers for each page: --- Page X --- (where X is the page number, starting from 1)
   - Page identifiers should be on their own line, with blank lines before and after
   - Example:
     
     --- Page 1 ---
     
     [Content from page 1]
     
     --- Page 2 ---
     
     [Content from page 2]
   
   - If processing multiple pages in one batch, add a separate identifier for each page at the start of that page's content
   - Always include page identifiers to maintain document structure and readability

8. Maintain the logical flow and structure of the document, ensuring that sections and subsections are properly formatted using markdown headers (# for main headers, ## for subheaders, etc.).

9. Use appropriate markdown syntax for other formatting elements such as bold, italic, lists, and code blocks as needed.

10. Return only the parsed content in markdown format, INCLUDING ALL IDENTIFIER TAGS ([TABLE], [IMAGE], [TOC], [HEADER]) and PAGE IDENTIFIERS (--- Page X ---) as specified above."""


# ============================================================================
# Image Description Prompts
# ============================================================================

IMAGE_DESCRIPTION_PROMPT = """Analyze this image from a document and provide a detailed, accurate description. 
This image appears to be important content (chart, graph, diagram, KPI visualization, flowchart, etc.).

Focus on:
- Content and subject matter
- Text visible in the image (if any) - include all numbers, labels, and text
- Charts, graphs, diagrams, or visual elements - describe the type and data
- Context that would help understand the image in the document
- Any important details, numbers, labels, axes, legends, or annotations
- For charts/graphs: describe the data trends, values, and what the visualization shows
- For flowcharts/diagrams: describe the structure, connections, and key elements

Provide a clear, concise, comprehensive description suitable for inclusion in a document markdown. 
Be thorough - include all visible text and data."""


PAGE_PROCESSING_PROMPT = """You are performing OCR, generating image descriptions, and creating markdown for a document page - ALL IN ONE STEP.

CRITICAL: You must maintain the EXACT layout, reading order, and positions as they appear on the page.

TASK:
1. Extract ALL text from the page (OCR) in markdown format
2. For each chart, graph, diagram, KPI, or important image, generate a detailed description
3. Place image descriptions at the EXACT position where each image appears in the document flow
4. Maintain the EXACT reading order (top to bottom, left to right)
5. Preserve spacing, structure, and visual layout

OUTPUT REQUIREMENTS:
- Generate complete markdown for the entire page
- Maintain the exact sequence: if text comes before an image, keep that order
- If an image appears between paragraphs, place its description between those paragraphs
- Use markdown headings (# ## ###) for titles, maintaining hierarchy
- Format tables as markdown tables (| col1 | col2 |)
- For each important image/chart, insert at its exact position: [IMAGE_DESCRIPTION: <detailed description>]
- Include ALL visible text, numbers, labels, data, annotations
- For charts: describe data, trends, axes, legends, values, insights
- For flowcharts: describe structure, connections, flow, elements
- Maintain blank lines and spacing to preserve visual structure

LAYOUT PRESERVATION:
- Read from top to bottom, left to right (exactly as a human would read)
- If image is at top, description goes at top
- If image is in middle, description goes in middle
- If image is at bottom, description goes at bottom
- Preserve relative positions: if image is between paragraph 2 and 3, place description there
- Maintain column layouts if present
- Keep the exact same reading flow as the original

Be thorough, accurate, and maintain exact positions. Output the complete page markdown:"""


# ============================================================================
# Table Processing Prompts
# ============================================================================

TABLE_EXTRACTION_PROMPT = """Extract the table from this image and format it as a markdown table.

Requirements:
- Extract ALL text, numbers, and data from the table
- Maintain the exact structure (rows and columns)
- Format as a proper markdown table with | separators
- Include headers if present
- Preserve all cell content accurately
- Output ONLY the markdown table, nothing else

Provide the table in markdown format:"""


TABLE_FORMATTING_PROMPT_TEMPLATE = """You are an expert in markdown tables. Transform the following parsed table into a clean markdown table. Provide just the table in pure markdown, nothing else.

<TEXT>
{table_text}
</TEXT>
"""


def get_table_formatting_prompt(table_text: str, previous_table: Optional[str] = None) -> str:
    """
    Create LLM prompt for table formatting.
    
    Args:
        table_text: The table text to format
        previous_table: Optional previous table for context
        
    Returns:
        Complete prompt string
    """
    prompt = TABLE_FORMATTING_PROMPT_TEMPLATE
    if previous_table:
        prompt += f"\n<PREVIOUS_TABLE>\n{previous_table}\n</PREVIOUS_TABLE>"
    
    return prompt.format(table_text=table_text)
