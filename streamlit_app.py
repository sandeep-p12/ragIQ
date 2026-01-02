"""Unified Streamlit UI for RAG IQ - Complete RAG pipeline."""

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Add project root to Python path
_project_root = Path(__file__).parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from centralized modules
from src.config.chunking import ChunkConfig
from src.config.parsing import ParseForgeConfig
from src.config.parsing_strategies import StrategyEnum
from src.config.retrieval import RetrievalConfig
from src.pipelines.chunking.chunking import process_document
from src.pipelines.chunking.retrieval_safety import expand_neighbors
from src.pipelines.parsing.parseforge import ParseForge
from src.pipelines.retrieval.retrieval import ingest_from_chunking_outputs, retrieve
from src.schema.chunk import Chunk, ParentChunk
from src.schema.document import Document
from src.utils.env import get_openai_api_key, get_azure_openai_endpoint, get_azure_openai_api_version, load_env
from src.utils.io import format_citation, load_jsonl, save_jsonl
from src.utils.ui import format_stage_output

logger = logging.getLogger(__name__)


def display_azure_openai_config(config: ParseForgeConfig, key_prefix: str = "azure_config"):
    """Display Azure OpenAI configuration details."""
    if config.llm_provider != "azure_openai":
        return
    
    st.markdown("---")
    st.markdown("**🔷 Azure OpenAI Configuration**")
    
    # Get values from config or environment
    endpoint = config.llm_azure_endpoint or get_azure_openai_endpoint()
    api_version = config.llm_azure_api_version or get_azure_openai_api_version()
    deployment_name = config.llm_azure_deployment_name or config.llm_model
    use_azure_ad = config.llm_use_azure_ad
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.text_input(
            "Endpoint",
            value=endpoint or "Not set",
            disabled=True,
            help="Azure OpenAI endpoint URL",
            key=f"{key_prefix}_endpoint"
        )
        st.text_input(
            "API Version",
            value=api_version or "Not set",
            disabled=True,
            help="Azure OpenAI API version",
            key=f"{key_prefix}_api_version"
        )
    
    with config_col2:
        st.text_input(
            "Deployment Name",
            value=deployment_name or "Not set",
            disabled=True,
            help="Azure OpenAI deployment name",
            key=f"{key_prefix}_deployment"
        )
        auth_method = "Azure AD (DefaultAzureCredential)" if use_azure_ad else "API Key"
        st.text_input(
            "Authentication",
            value=auth_method,
            disabled=True,
            help="Authentication method used",
            key=f"{key_prefix}_auth"
        )
    
    # Show model name
    st.text_input(
        "Model",
        value=config.llm_model or "Not set",
        disabled=True,
        help="LLM model name",
        key=f"{key_prefix}_model"
    )

# Configure page
st.set_page_config(
    page_title="RAG IQ",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if "document" not in st.session_state:
    st.session_state.document = None
if "parsed_markdown" not in st.session_state:
    st.session_state.parsed_markdown = None
if "parsed_file_name" not in st.session_state:
    st.session_state.parsed_file_name = None
if "chunks_children" not in st.session_state:
    st.session_state.chunks_children = []
if "chunks_parents" not in st.session_state:
    st.session_state.chunks_parents = []
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {}
if "retrieval_result" not in st.session_state:
    st.session_state.retrieval_result = None
if "current_stage" not in st.session_state:
    st.session_state.current_stage = ""
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
if "stage_outputs" not in st.session_state:
    st.session_state.stage_outputs = {}


def progress_callback(stage: str, progress: float, output_data: Optional[Dict] = None):
    """Progress callback for Streamlit."""
    st.session_state.current_stage = stage
    st.session_state.progress = progress
    if output_data is not None:
        formatted_output = format_stage_output(stage, output_data)
        st.session_state.stage_outputs[stage] = formatted_output


def main():
    """Main Streamlit app."""
    st.title("📚 RAG IQ - Complete RAG Pipeline")
    st.markdown("Production-grade RAG system with parsing, chunking, indexing, and retrieval")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Parse",
        "✂️ Chunk",
        "📊 Index",
        "🔍 Retrieve",
        "🔄 Pipeline"
    ])
    
    with tab1:
        render_parse_tab()
    
    with tab2:
        render_chunk_tab()
    
    with tab3:
        render_index_tab()
    
    with tab4:
        render_retrieve_tab()
    
    with tab5:
        render_pipeline_tab()


def render_parse_tab():
    """Render parsing tab."""
    st.header("Document Parsing")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Parsing Configuration")
        
        # Strategy
        strategy_mode = st.selectbox(
            "Parsing Strategy",
            ["auto", "fast", "hi_res", "llm_full"],
            index=0,
            help="AUTO: Automatically selects FAST or HI_RES per page. FAST: Native text extraction. HI_RES: OCR-based. LLM_FULL: Full document parsing using LLM vision."
        )
        
        # Device
        device = st.selectbox("Device", ["cpu", "cuda", "mps", "coreml"], index=0)
        
        # Processing
        with st.expander("Processing Settings", expanded=False):
            batch_size = st.slider("Batch Size", 10, 100, 50, help="Number of pages to process in parallel")
            page_threshold = st.slider("Page Threshold", 0.0, 1.0, 0.6, 0.1, help="IoU threshold for page-level strategy")
            document_threshold = st.slider("Document Threshold", 0.0, 1.0, 0.2, 0.1, help="Threshold for document-level strategy")
        
        # Finance Mode
        with st.expander("Finance Mode", expanded=False):
            finance_mode = st.checkbox("Enable Finance Mode", value=False, help="Stricter thresholds for financial documents")
            if finance_mode:
                finance_page_threshold = st.slider("Finance Page Threshold", 0.0, 1.0, 0.7, 0.1)
                finance_document_threshold = st.slider("Finance Document Threshold", 0.0, 1.0, 0.15, 0.1)
            else:
                finance_page_threshold = 0.7
                finance_document_threshold = 0.15
        
        # LLM Settings
        with st.expander("LLM Settings", expanded=False):
            # Load default from .env
            default_config = ParseForgeConfig()
            default_provider = default_config.llm_provider
            
            # Provider dropdown - only UI control, all other settings from .env
            provider_options = ["openai", "azure_openai", "none"]
            default_index = provider_options.index(default_provider) if default_provider in provider_options else 0
            llm_provider = st.selectbox(
                "LLM Provider", 
                provider_options, 
                index=default_index,
                help="Select provider. All other settings (endpoint, API key, etc.) are read from .env file"
            )
            
            # Show Azure OpenAI config if selected
            if llm_provider == "azure_openai":
                # Create config with selected provider to show current settings
                display_config = ParseForgeConfig(llm_provider=llm_provider)
                display_azure_openai_config(display_config, key_prefix="parse_azure_config")
        
        # Page Range
        with st.expander("Page Range", expanded=False):
            use_page_range = st.checkbox("Limit page range", value=False)
            start_page = st.number_input("Start Page", min_value=0, value=0, disabled=not use_page_range)
            end_page = st.number_input("End Page", min_value=1, value=100, disabled=not use_page_range)
        
        # Checkpoint
        with st.expander("Checkpoint Settings", expanded=False):
            auto_resume = st.checkbox("Auto Resume", value=True, help="Automatically resume from checkpoints")
    
    # Main content
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "html", "txt", "md"]
    )
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Create config from .env, only override provider from UI
        config = ParseForgeConfig(
            device=device,
            batch_size=batch_size,
            page_threshold=page_threshold,
            document_threshold=document_threshold,
            finance_mode=finance_mode,
            finance_page_threshold=finance_page_threshold if finance_mode else 0.7,
            finance_document_threshold=finance_document_threshold if finance_mode else 0.15,
            llm_provider=llm_provider,  # Only override provider from UI, rest from .env
            auto_resume=auto_resume,
        )
        
        if st.button("Parse Document", type="primary"):
            try:
                parser = ParseForge(config, progress_callback=progress_callback)
                
                # Map UI selection to StrategyEnum
                strategy_map = {
                    "auto": StrategyEnum.AUTO,
                    "fast": StrategyEnum.FAST,
                    "hi_res": StrategyEnum.HI_RES,
                    "llm_full": StrategyEnum.LLM_FULL,
                }
                strategy = strategy_map[strategy_mode]
                
                if strategy == StrategyEnum.LLM_FULL and (config.llm_provider == "none" or not config.llm_api_key):
                    st.warning("⚠️ LLM_FULL strategy requires LLM configuration. Please set PARSEFORGE_LLM_API_KEY in your .env file.")
                
                # Progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Enhanced progress callback that updates UI
                def ui_progress_callback(stage: str, progress: float, output_data: Optional[Dict] = None):
                    progress_callback(stage, progress, output_data)
                    progress_bar.progress(min(1.0, max(0.0, progress)))
                    status_msg = f"🔄 {stage} ({progress*100:.1f}%)"
                    if output_data:
                        try:
                            details = format_stage_output(stage, output_data)
                            if isinstance(details, dict):
                                # Extract key info from dict
                                detail_str = ", ".join([f"{k}: {v}" for k, v in details.items() if v])
                                if detail_str:
                                    status_msg += f" - {detail_str}"
                            elif details:
                                status_msg += f" - {details}"
                        except Exception:
                            pass  # Ignore formatting errors
                    status_text.text(status_msg)
                
                # Parse with progress
                parser_with_progress = ParseForge(config, progress_callback=ui_progress_callback)
                
                with st.spinner("Parsing document..."):
                    document = parser_with_progress.parse(
                        tmp_path,
                        strategy=strategy,
                        start_page=start_page if use_page_range else None,
                        end_page=end_page if use_page_range else None,
                    )
                    st.session_state.document = document
                
                progress_bar.progress(0.9)
                status_text.text("🔄 Generating markdown...")
                
                st.success("Document parsed successfully!")
                
                # Generate markdown
                with st.spinner("Generating markdown..."):
                    generate_image_descriptions = config.llm_provider != "none" and config.llm_api_key is not None
                    markdown = parser_with_progress.to_markdown(document, generate_image_descriptions=generate_image_descriptions)
                    st.session_state.parsed_markdown = markdown
                    st.session_state.parsed_file_name = uploaded_file.name
                
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                status_text.empty()
                
                # Display Azure OpenAI config if used
                if config.llm_provider == "azure_openai":
                    with st.expander("🔷 Azure OpenAI Configuration Used", expanded=False):
                        display_azure_openai_config(config, key_prefix="parse_result_azure")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Document Info")
                    st.json({
                        "pages": len(document.pages),
                        "total_blocks": sum(len(page.blocks) for page in document.pages),
                        "markdown_length": len(markdown),
                        "markdown_lines": len(markdown.splitlines()),
                    })
                with col2:
                    st.subheader("Download")
                    import json
                    json_data = json.dumps(document.model_dump(), indent=2)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        file_name="parsed.json",
                        mime="application/json",
                    )
                    st.download_button(
                        "Download Markdown",
                        markdown,
                        file_name=f"{Path(uploaded_file.name).stem}.md",
                        mime="text/markdown",
                    )
                
            except Exception as e:
                st.error(f"Error parsing document: {e}")
                logger.exception(e)
    
    # Show markdown preview if available (persists across tab switches)
    if st.session_state.parsed_markdown and st.session_state.parsed_file_name:
        st.divider()
        st.subheader("📝 Markdown Preview")
        preview_tabs = st.tabs(["Preview", "Raw Markdown", "Statistics"])
        
        markdown = st.session_state.parsed_markdown
        document = st.session_state.document
        
        with preview_tabs[0]:
            st.markdown(markdown)
        
        with preview_tabs[1]:
            st.code(markdown, language="markdown")
        
        with preview_tabs[2]:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Characters", f"{len(markdown):,}")
            with col2:
                st.metric("Total Lines", f"{len(markdown.splitlines()):,}")
            with col3:
                st.metric("Pages", len(document.pages) if document else 0)
            with col4:
                total_blocks = sum(len(page.blocks) for page in document.pages) if document else 0
                st.metric("Total Blocks", total_blocks)
            
            if document:
                # Block type distribution
                block_types = {}
                for page in document.pages:
                    for block in page.blocks:
                        block_type = block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type)
                        block_types[block_type] = block_types.get(block_type, 0) + 1
                
                st.subheader("Block Type Distribution")
                st.bar_chart(block_types)


def render_chunk_tab():
    """Render chunking tab."""
    st.header("Document Chunking")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Chunking Configuration")
        
        # Prose chunking
        with st.expander("Prose Chunking", expanded=True):
            prose_target = st.slider(
                "Prose Target Tokens",
                min_value=128,
                max_value=2048,
                value=512,
                step=64,
                help="Target token size for prose chunks"
            )
            prose_overlap = st.slider(
                "Prose Overlap Tokens",
                min_value=0,
                max_value=256,
                value=50,
                step=8,
                help="Overlap tokens between prose chunks"
            )
            sentence_window = st.slider(
                "Sentence Window Size",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Number of sentences to consider for windowing"
            )
        
        # List chunking
        with st.expander("List Chunking", expanded=False):
            list_items = st.slider(
                "List Items Per Chunk",
                min_value=5,
                max_value=50,
                value=10,
                step=1
            )
            list_overlap = st.slider(
                "List Item Overlap",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )
        
        # Table chunking
        with st.expander("Table Chunking", expanded=False):
            table_rows = st.slider(
                "Table Rows Per Chunk",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
            table_overlap = st.slider(
                "Table Row Overlap",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )
        
        # Hierarchy
        with st.expander("Hierarchy", expanded=False):
            parent_heading_level = st.selectbox(
                "Parent Heading Level",
                options=[1, 2, 3],
                index=1,  # H2
                help="Heading level for parent grouping"
            )
            parent_page_window = st.slider(
                "Parent Page Window Size",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Page window size for soft section grouping"
            )
            confidence_threshold = st.slider(
                "Structure Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Threshold for heading-based vs soft-section grouping"
            )
        
        # Neighbors
        with st.expander("Retrieval Neighbors", expanded=False):
            neighbor_same = st.slider(
                "Same Page Neighbors",
                min_value=0,
                max_value=5,
                value=1,
                step=1,
                help="Number of sibling neighbors within same parent"
            )
            neighbor_cross = st.slider(
                "Cross Page Neighbors",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                help="Number of neighbors from adjacent pages"
            )
        
        # General
        with st.expander("General Settings", expanded=False):
            max_tokens = st.slider(
                "Max Chunk Tokens (Hard Limit)",
                min_value=512,
                max_value=4096,
                value=2048,
                step=256
            )
            min_tokens = st.slider(
                "Min Chunk Tokens",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Minimum target size for chunks"
            )
            enable_merge = st.checkbox(
                "Enable Cross-Page Merge",
                value=True
            )
            merge_aggressiveness = st.selectbox(
                "Merge Aggressiveness",
                options=["low", "medium", "high"],
                index=1  # medium
            )
    
    # Main content
    # Check if there's a parsed document available
    use_parsed_document = False
    tmp_path = None
    file_name = None
    
    if st.session_state.parsed_markdown and st.session_state.parsed_file_name:
        st.info(f"📄 Parsed document available: **{st.session_state.parsed_file_name}**")
        use_parsed = st.checkbox(
            f"Use parsed document: {st.session_state.parsed_file_name}",
            value=True,
            key="use_parsed_doc"
        )
        if use_parsed:
            use_parsed_document = True
            st.success(f"✅ Will use parsed markdown from: {st.session_state.parsed_file_name}")
    
    uploaded_file = st.file_uploader(
        "Or upload a Markdown file",
        type=["md", "markdown"],
        key="chunk_upload",
        disabled=use_parsed_document
    )
    
    # Determine which file to use
    if use_parsed_document and st.session_state.parsed_markdown:
        # Use parsed markdown
        markdown_content = st.session_state.parsed_markdown
        file_name = st.session_state.parsed_file_name or "parsed_document.md"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(markdown_content)
            tmp_path = tmp_file.name
    elif uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(uploaded_file.getvalue().decode("utf-8"))
            tmp_path = tmp_file.name
        file_name = uploaded_file.name
    
    if not tmp_path:
        st.warning("Please either use a parsed document from the Parse tab or upload a markdown file.")
        return
    
    if tmp_path:
        # Create config from UI settings
        config = ChunkConfig(
            prose_target_tokens=prose_target,
            prose_overlap_tokens=prose_overlap,
            sentence_window_size=sentence_window,
            list_items_per_chunk=list_items,
            list_item_overlap=list_overlap,
            table_rows_per_chunk=table_rows,
            table_row_overlap=table_overlap,
            parent_heading_level=parent_heading_level,
            parent_page_window_size=parent_page_window,
            neighbor_same_page=neighbor_same,
            neighbor_cross_page=neighbor_cross,
            max_chunk_tokens_hard=max_tokens,
            min_chunk_tokens=min_tokens,
            enable_cross_page_merge=enable_merge,
            cross_page_merge_aggressiveness=merge_aggressiveness,
            structure_confidence_threshold=confidence_threshold,
        )
        
        # Show file info
        if use_parsed_document:
            st.info(f"📄 Using parsed document: **{file_name}** ({len(markdown_content):,} characters)")
        else:
            st.info(f"📄 Using uploaded file: **{file_name}**")
        
        doc_id = st.text_input("Document ID", value=Path(file_name).stem if file_name else "test_doc")
        
        if st.button("Chunk Document", type="primary"):
            try:
                with st.spinner("Chunking document..."):
                    children, parents, stats = process_document(tmp_path, config, doc_id)
                    st.session_state.chunks_children = children
                    st.session_state.chunks_parents = parents
                    st.session_state.processing_stats = stats
                
                st.success(f"Chunking complete! {len(children)} children, {len(parents)} parents")
                
                # Display stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Children Chunks", len(children))
                with col2:
                    st.metric("Parent Chunks", len(parents))
                with col3:
                    avg_tokens = stats.get("avg_tokens_per_chunk", 0)
                    st.metric("Avg Tokens/Chunk", f"{avg_tokens:.0f}")
                
                # Export chunks
                st.subheader("📤 Export Chunks")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # Export children chunks
                    children_json = json.dumps([chunk.__dict__ for chunk in children], indent=2, default=str)
                    st.download_button(
                        "Download Children Chunks (JSON)",
                        children_json,
                        file_name=f"{doc_id}_children.json",
                        mime="application/json",
                    )
                
                with export_col2:
                    # Export parent chunks
                    parents_json = json.dumps([parent.__dict__ for parent in parents], indent=2, default=str)
                    st.download_button(
                        "Download Parent Chunks (JSON)",
                        parents_json,
                        file_name=f"{doc_id}_parents.json",
                        mime="application/json",
                    )
                
                # Export as JSONL
                st.download_button(
                    "Download All Chunks (JSONL)",
                    "\n".join([json.dumps(chunk.__dict__, default=str) for chunk in children + parents]),
                    file_name=f"{doc_id}_chunks.jsonl",
                    mime="application/jsonl",
                )
                
                # Chunk preview
                st.subheader("📋 Chunk Preview")
                preview_tabs = st.tabs(["Children Chunks", "Parent Chunks", "Statistics"])
                
                with preview_tabs[0]:
                    st.write(f"**Total Children Chunks: {len(children)}**")
                    for i, chunk in enumerate(children[:10], 1):  # Show first 10
                        with st.expander(f"Chunk {i}: {chunk.chunk_id[:30]}...", expanded=(i <= 3)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Metadata:**")
                                st.json({
                                    "chunk_id": chunk.chunk_id,
                                    "doc_id": chunk.doc_id,
                                    "element_type": chunk.element_type if hasattr(chunk, 'element_type') else 'unknown',
                                    "token_count": chunk.token_count if hasattr(chunk, 'token_count') else 0,
                                    "page_nos": chunk.page_nos if hasattr(chunk, 'page_nos') else [],
                                    "section_label": chunk.section_label if hasattr(chunk, 'section_label') else '',
                                })
                            with col2:
                                st.write("**Text for Embedding:**")
                                st.text_area("", chunk.text_for_embedding[:500] if chunk.text_for_embedding else "", height=200, key=f"chunk_preview_child_text_{i}_{chunk.chunk_id[:20]}", disabled=True)
                    if len(children) > 10:
                        st.info(f"Showing first 10 of {len(children)} children chunks")
                
                with preview_tabs[1]:
                    st.write(f"**Total Parent Chunks: {len(parents)}**")
                    for i, parent in enumerate(parents[:10], 1):  # Show first 10
                        with st.expander(f"Parent {i}: {parent.chunk_id[:30]}...", expanded=(i <= 3)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Metadata:**")
                                st.json({
                                    "chunk_id": parent.chunk_id,
                                    "doc_id": parent.doc_id,
                                    "parent_type": parent.parent_type if hasattr(parent, 'parent_type') else 'unknown',
                                    "child_ids": parent.child_ids if hasattr(parent, 'child_ids') else [],
                                    "num_children": len(parent.child_ids) if hasattr(parent, 'child_ids') else 0,
                                    "page_nos": parent.page_nos if hasattr(parent, 'page_nos') else [],
                                })
                            with col2:
                                st.write("**Text for Embedding:**")
                                st.text_area("", parent.text_for_embedding[:500] if parent.text_for_embedding else "", height=200, key=f"chunk_preview_parent_text_{i}_{parent.chunk_id[:20]}", disabled=True)
                    if len(parents) > 10:
                        st.info(f"Showing first 10 of {len(parents)} parent chunks")
                
                with preview_tabs[2]:
                    st.json(stats)
                    
                    # Token distribution
                    token_counts = [chunk.token_count for chunk in children if hasattr(chunk, 'token_count') and chunk.token_count]
                    if token_counts:
                        st.subheader("Token Distribution")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Min Tokens", min(token_counts))
                        with col2:
                            st.metric("Max Tokens", max(token_counts))
                        with col3:
                            st.metric("Avg Tokens", f"{sum(token_counts) / len(token_counts):.0f}")
                        
                        # Histogram
                        import pandas as pd
                        df = pd.DataFrame({"tokens": token_counts})
                        st.bar_chart(df)
                    
                    # Element type distribution
                    element_types = {}
                    for chunk in children:
                        elem_type = chunk.element_type if hasattr(chunk, 'element_type') else 'unknown'
                        element_types[elem_type] = element_types.get(elem_type, 0) + 1
                    
                    if element_types:
                        st.subheader("Element Type Distribution")
                        st.bar_chart(element_types)
            except Exception as e:
                st.error(f"Error chunking document: {e}")
                logger.exception(e)
    
    # Show chunk preview if available (persists across tab switches)
    if st.session_state.chunks_children and st.session_state.chunks_parents:
        st.divider()
        st.subheader("📋 Chunk Preview & Filters")
        
        # Filter controls
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            element_types = list(set([chunk.element_type if hasattr(chunk, 'element_type') else 'unknown' 
                                     for chunk in st.session_state.chunks_children]))
            element_types.sort()
            selected_element_types = st.multiselect(
                "Filter by Element Type",
                options=element_types,
                default=element_types,
                key="chunk_filter_element"
            )
        
        with filter_col2:
            # Page range filter
            all_pages = set()
            for chunk in st.session_state.chunks_children:
                if hasattr(chunk, 'page_nos') and chunk.page_nos:
                    all_pages.update(chunk.page_nos)
            if all_pages:
                min_page, max_page = min(all_pages), max(all_pages)
                page_range = st.slider(
                    "Filter by Page Range",
                    min_value=int(min_page),
                    max_value=int(max_page),
                    value=(int(min_page), int(max_page)),
                    key="chunk_filter_page"
                )
            else:
                page_range = None
        
        with filter_col3:
            # Section label filter
            section_labels = list(set([chunk.section_label if hasattr(chunk, 'section_label') and chunk.section_label else 'Unknown'
                                      for chunk in st.session_state.chunks_children]))
            section_labels.sort()
            selected_sections = st.multiselect(
                "Filter by Section",
                options=section_labels,
                default=section_labels,
                key="chunk_filter_section"
            )
        
        with filter_col4:
            # Token range filter
            token_counts = [chunk.token_count for chunk in st.session_state.chunks_children 
                          if hasattr(chunk, 'token_count') and chunk.token_count]
            if token_counts:
                min_tokens, max_tokens = min(token_counts), max(token_counts)
                token_range = st.slider(
                    "Filter by Token Range",
                    min_value=int(min_tokens),
                    max_value=int(max_tokens),
                    value=(int(min_tokens), int(max_tokens)),
                    key="chunk_filter_tokens"
                )
            else:
                token_range = None
        
        # Apply filters
        filtered_children = st.session_state.chunks_children
        filtered_parents = st.session_state.chunks_parents
        
        if selected_element_types:
            filtered_children = [c for c in filtered_children 
                               if (hasattr(c, 'element_type') and c.element_type in selected_element_types) 
                               or (not hasattr(c, 'element_type') and 'unknown' in selected_element_types)]
        
        if page_range:
            filtered_children = [c for c in filtered_children
                               if hasattr(c, 'page_nos') and c.page_nos 
                               and any(page_range[0] <= p <= page_range[1] for p in c.page_nos)]
        
        if selected_sections:
            filtered_children = [c for c in filtered_children
                               if (hasattr(c, 'section_label') and c.section_label in selected_sections)
                               or (not hasattr(c, 'section_label') and 'Unknown' in selected_sections)]
        
        if token_range:
            filtered_children = [c for c in filtered_children
                               if hasattr(c, 'token_count') and c.token_count
                               and token_range[0] <= c.token_count <= token_range[1]]
        
        # Show filtered count
        st.info(f"Showing {len(filtered_children)} of {len(st.session_state.chunks_children)} children chunks")
        
        preview_tabs = st.tabs(["Children Chunks", "Parent Chunks", "Statistics"])
        
        with preview_tabs[0]:
            st.write(f"**Total Children Chunks: {len(filtered_children)}**")
            for i, chunk in enumerate(filtered_children[:20], 1):  # Show first 20 filtered
                with st.expander(f"Chunk {i}: {chunk.chunk_id[:30]}...", expanded=(i <= 3)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Metadata:**")
                        st.json({
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "element_type": chunk.element_type if hasattr(chunk, 'element_type') else 'unknown',
                            "token_count": chunk.token_count if hasattr(chunk, 'token_count') else 0,
                            "page_nos": chunk.page_nos if hasattr(chunk, 'page_nos') else [],
                            "section_label": chunk.section_label if hasattr(chunk, 'section_label') else '',
                        })
                    with col2:
                        st.write("**Text for Embedding:**")
                        st.text_area("", chunk.text_for_embedding[:500] if chunk.text_for_embedding else "", 
                                   height=200, key=f"chunk_filtered_child_text_{i}_{chunk.chunk_id[:20]}", disabled=True)
            if len(filtered_children) > 20:
                st.info(f"Showing first 20 of {len(filtered_children)} filtered children chunks")
        
        with preview_tabs[1]:
            st.write(f"**Total Parent Chunks: {len(filtered_parents)}**")
            for i, parent in enumerate(filtered_parents[:20], 1):  # Show first 20
                with st.expander(f"Parent {i}: {parent.chunk_id[:30]}...", expanded=(i <= 3)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Metadata:**")
                        st.json({
                            "chunk_id": parent.chunk_id,
                            "doc_id": parent.doc_id,
                            "parent_type": parent.parent_type if hasattr(parent, 'parent_type') else 'unknown',
                            "child_ids": parent.child_ids if hasattr(parent, 'child_ids') else [],
                            "num_children": len(parent.child_ids) if hasattr(parent, 'child_ids') else 0,
                            "page_nos": parent.page_nos if hasattr(parent, 'page_nos') else [],
                        })
                    with col2:
                        st.write("**Text for Embedding:**")
                        st.text_area("", parent.text_for_embedding[:500] if parent.text_for_embedding else "", 
                                   height=200, key=f"chunk_filtered_parent_text_{i}_{parent.chunk_id[:20]}", disabled=True)
            if len(filtered_parents) > 20:
                st.info(f"Showing first 20 of {len(filtered_parents)} parent chunks")
        
        with preview_tabs[2]:
            if st.session_state.processing_stats:
                st.json(st.session_state.processing_stats)
            
            # Token distribution
            token_counts = [chunk.token_count for chunk in filtered_children 
                          if hasattr(chunk, 'token_count') and chunk.token_count]
            if token_counts:
                st.subheader("Token Distribution")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Tokens", min(token_counts))
                with col2:
                    st.metric("Max Tokens", max(token_counts))
                with col3:
                    st.metric("Avg Tokens", f"{sum(token_counts) / len(token_counts):.0f}")
                
                # Histogram
                import pandas as pd
                df = pd.DataFrame({"tokens": token_counts})
                st.bar_chart(df)
            
            # Element type distribution
            element_types = {}
            for chunk in filtered_children:
                elem_type = chunk.element_type if hasattr(chunk, 'element_type') else 'unknown'
                element_types[elem_type] = element_types.get(elem_type, 0) + 1
            
            if element_types:
                st.subheader("Element Type Distribution")
                st.bar_chart(element_types)


def render_index_tab():
    """Render indexing tab."""
    st.header("Index to Vector Store")
    
    if not st.session_state.chunks_children:
        st.warning("Please chunk a document first in the Chunk tab.")
        return
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Indexing Configuration")
        
        # Embedding config
        with st.expander("Embedding Settings", expanded=True):
            # Load default from .env
            from src.config.retrieval import EmbeddingConfig
            default_embedding_config = EmbeddingConfig.from_env()
            default_embedding_provider = default_embedding_config.provider
            
            # Provider dropdown - only UI control, all other settings from .env
            embedding_provider_options = ["openai", "azure_openai"]
            default_embedding_index = embedding_provider_options.index(default_embedding_provider) if default_embedding_provider in embedding_provider_options else 0
            embedding_provider = st.selectbox(
                "Embedding Provider", 
                embedding_provider_options, 
                index=default_embedding_index,
                help="Select provider. All other settings (endpoint, API key, deployment name, etc.) are read from .env file"
            )
            
            embedding_batch_size = st.slider(
                "Embedding Batch Size",
                min_value=10,
                max_value=500,
                value=default_embedding_config.batch_size,
                step=10,
                help="Number of texts to embed in parallel"
            )
            embedding_max_retries = st.number_input(
                "Max Retries",
                min_value=1,
                max_value=10,
                value=default_embedding_config.max_retries,
                step=1
            )
        
        # Pinecone config
        with st.expander("Pinecone Settings", expanded=False):
            index_name = st.text_input(
                "Index Name",
                value="hybrid-chunking",
                help="Pinecone index name"
            )
            namespace = st.selectbox(
                "Namespace",
                options=["children", "parents"],
                index=0,
                help="Pinecone namespace"
            )
            top_k_dense = st.slider(
                "Top K Dense",
                min_value=10,
                max_value=500,
                value=300,
                step=10,
                help="Number of dense vectors to retrieve"
            )
    
    doc_id = st.text_input("Document ID", value="test_doc")
    
    if st.button("Index Chunks", type="primary"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as children_file:
                for chunk in st.session_state.chunks_children:
                    children_file.write(json.dumps(chunk.__dict__) + "\n")
                children_path = children_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as parents_file:
                for chunk in st.session_state.chunks_parents:
                    parents_file.write(json.dumps(chunk.__dict__) + "\n")
                parents_path = parents_file.name
            
            # Create config from .env, only override provider and batch settings from UI
            from src.config.retrieval import EmbeddingConfig, PineconeConfig
            embedding_config = EmbeddingConfig.from_env(
                provider=embedding_provider,  # Only override provider from UI, rest from .env
                batch_size=embedding_batch_size,
                max_retries=embedding_max_retries,
            )
            pinecone_config = PineconeConfig.from_env(
                namespace=namespace,
                top_k_dense=top_k_dense,
            )
            # Override index_name after initialization
            pinecone_config.index_name = index_name
            
            cfg = RetrievalConfig.from_env()
            cfg.embedding_config = embedding_config
            cfg.pinecone_config = pinecone_config
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Status: Starting indexing...")
            progress_bar.progress(0.1)
            
            with st.spinner("Indexing chunks..."):
                stats = ingest_from_chunking_outputs(children_path, parents_path, doc_id, cfg)
            
            progress_bar.progress(1.0)
            status_text.text("Status: Complete!")
            st.success("Indexing complete!")
            
            # Display stats
            if isinstance(stats, dict):
                st.subheader("📊 Indexing Statistics")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chunks Indexed", stats.get("chunks_indexed", 0))
                with col2:
                    st.metric("Vectors Created", stats.get("vectors_created", 0))
                with col3:
                    st.metric("Time (s)", f"{stats.get('time_seconds', 0):.2f}")
                with col4:
                    st.metric("Index Name", index_name)
                
                # Detailed breakdown
                st.subheader("📋 Detailed Breakdown")
                
                if "embedding_stats" in stats:
                    st.write("**Embedding Statistics:**")
                    st.json(stats["embedding_stats"])
                
                if "pinecone_stats" in stats:
                    st.write("**Pinecone Statistics:**")
                    st.json(stats["pinecone_stats"])
                
                # Show all stats
                with st.expander("Complete Statistics", expanded=False):
                    st.json(stats)
                
                # Show indexed chunks preview
                st.subheader("📝 Indexed Chunks Preview")
                if st.session_state.chunks_children:
                    preview_count = st.slider("Number of chunks to preview", 1, min(20, len(st.session_state.chunks_children)), 5, key="index_preview_count")
                    for i, chunk in enumerate(st.session_state.chunks_children[:preview_count], 1):
                        with st.expander(f"Chunk {i}: {chunk.chunk_id[:40]}...", expanded=(i <= 2)):
                            st.write("**Chunk ID:**", chunk.chunk_id)
                            st.write("**Element Type:**", chunk.element_type if hasattr(chunk, 'element_type') else 'unknown')
                            st.write("**Pages:**", chunk.page_nos if hasattr(chunk, 'page_nos') else [])
                            st.write("**Section:**", chunk.section_label if hasattr(chunk, 'section_label') else '')
                            st.write("**Text Preview:**")
                            st.text(chunk.text_for_embedding[:300] if chunk.text_for_embedding else "")
            else:
                st.json(stats)
            
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Error indexing: {e}")
            logger.exception(e)


def render_retrieve_tab():
    """Render retrieval tab."""
    st.header("Query and Retrieve")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Retrieval Configuration")
        
        # Embedding config
        with st.expander("Embedding Settings", expanded=False):
            # Load default from .env
            from src.config.retrieval import EmbeddingConfig
            default_embedding_config = EmbeddingConfig.from_env()
            default_embedding_provider = default_embedding_config.provider
            
            # Provider dropdown - only UI control, all other settings from .env
            embedding_provider_options = ["openai", "azure_openai"]
            default_embedding_index = embedding_provider_options.index(default_embedding_provider) if default_embedding_provider in embedding_provider_options else 0
            embedding_provider = st.selectbox(
                "Embedding Provider",
                embedding_provider_options,
                index=default_embedding_index,
                key="retrieve_embedding_provider",
                help="Select provider. All other settings (endpoint, API key, deployment name, etc.) are read from .env file"
            )
        
        # Reranking config
        with st.expander("Reranking Settings", expanded=True):
            # Load default from .env
            default_rerank_config = ParseForgeConfig()
            default_rerank_provider = default_rerank_config.llm_provider
            
            # Provider dropdown - only UI control, all other settings from .env
            rerank_provider_options = ["openai", "azure_openai"]
            default_rerank_index = rerank_provider_options.index(default_rerank_provider) if default_rerank_provider in rerank_provider_options else 0
            rerank_provider = st.selectbox(
                "Rerank LLM Provider",
                rerank_provider_options,
                index=default_rerank_index,
                help="Select provider. All other settings (endpoint, API key, deployment name, model, etc.) are read from .env file",
                key="rerank_provider"
            )
            
            # Load rerank config from .env
            from src.config.retrieval import RerankConfig
            default_rerank_cfg = RerankConfig.from_env()
            rerank_model = st.text_input(
                "Rerank Model",
                value=default_rerank_cfg.model,
                help="LLM model or Azure deployment name for reranking (from .env: RERANK_MODEL or PARSEFORGE_LLM_MODEL)",
                key="rerank_model"
            )
            
            # Show Azure OpenAI config if selected
            if rerank_provider == "azure_openai":
                # Create config with selected provider and model to show current settings
                display_config = ParseForgeConfig(
                    llm_provider=rerank_provider,
                    llm_model=rerank_model
                )
                display_azure_openai_config(display_config, key_prefix="retrieve_rerank_azure_config")
            max_candidates = st.slider(
                "Max Candidates to Rerank",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Maximum number of candidates to send to LLM"
            )
            return_top_n = st.slider(
                "Return Top N",
                min_value=5,
                max_value=50,
                value=15,
                step=1,
                help="Final number of chunks in context"
            )
            max_text_chars = st.slider(
                "Max Text Chars Per Candidate",
                min_value=500,
                max_value=2000,
                value=1200,
                step=100,
                help="Maximum characters per candidate text"
            )
            strict_json = st.checkbox(
                "Strict JSON Output",
                value=True,
                help="Enforce strict JSON output from LLM"
            )
        
        # Retrieval config
        with st.expander("Retrieval Settings", expanded=False):
            neighbor_same = st.slider(
                "Same Page Neighbors",
                min_value=0,
                max_value=5,
                value=1,
                step=1,
                key="retrieve_neighbor_same"
            )
            neighbor_cross = st.slider(
                "Cross Page Neighbors",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                key="retrieve_neighbor_cross"
            )
            include_parents = st.checkbox(
                "Include Parents",
                value=True,
                help="Include parent chunks in context assembly"
            )
            final_max_tokens = st.slider(
                "Final Max Tokens",
                min_value=2000,
                max_value=20000,
                value=12000,
                step=1000,
                help="Token budget for final context pack"
            )
            min_primary_hits = st.number_input(
                "Min Primary Hits to Keep",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Minimum number of primary chunks to always keep"
            )
        
        # Pinecone config
        with st.expander("Pinecone Settings", expanded=False):
            index_name = st.text_input(
                "Index Name",
                value="hybrid-chunking",
                key="retrieve_index_name"
            )
            namespace = st.selectbox(
                "Namespace",
                options=["children", "parents"],
                index=0,
                key="retrieve_namespace"
            )
            top_k_dense = st.slider(
                "Top K Dense",
                min_value=10,
                max_value=500,
                value=300,
                step=10,
                key="retrieve_top_k"
            )
    
    query = st.text_area("Enter your query", height=100)
    
    if st.button("Retrieve", type="primary"):
        try:
            # Create config from .env, only override provider from UI
            from src.config.retrieval import EmbeddingConfig, PineconeConfig, RerankConfig
            embedding_config = EmbeddingConfig.from_env(
                provider=embedding_provider,  # Only override provider from UI, rest from .env
            )
            rerank_config = RerankConfig.from_env(
                model=rerank_model,  # Allow model override from UI
                max_candidates_to_rerank=max_candidates,
                return_top_n=return_top_n,
                max_text_chars_per_candidate=max_text_chars,
                strict_json_output=strict_json,
            )
            pinecone_config = PineconeConfig.from_env(
                namespace=namespace,
                top_k_dense=top_k_dense,
            )
            # Override index_name after initialization
            pinecone_config.index_name = index_name
            
            cfg = RetrievalConfig.from_env()
            cfg.embedding_config = embedding_config
            cfg.rerank_config = rerank_config
            cfg.pinecone_config = pinecone_config
            cfg.neighbor_same_page = neighbor_same
            cfg.neighbor_cross_page = neighbor_cross
            cfg.include_parents = include_parents
            cfg.final_max_tokens = final_max_tokens
            cfg.min_primary_hits_to_keep = min_primary_hits
            
            # Create LLM config for reranker from .env, only override provider from UI
            # ParseForgeConfig is already imported at the top of the file
            rerank_llm_config = ParseForgeConfig(
                llm_provider=rerank_provider,  # Only override provider from UI, rest from .env
                llm_model=rerank_model,  # Allow model override from UI
            )
            
            all_chunks = [chunk.__dict__ for chunk in st.session_state.chunks_children]
            
            with st.spinner("Retrieving..."):
                result = retrieve(query, {}, cfg, all_chunks, llm_config=rerank_llm_config)
            
            st.success("Retrieval complete!")
            
            # Store result in session state for persistence
            st.session_state.retrieval_result = result
            
            # Display Azure OpenAI config if used for reranking
            if rerank_provider == "azure_openai":
                with st.expander("🔷 Azure OpenAI Configuration Used (Reranking)", expanded=False):
                    display_config = ParseForgeConfig(
                        llm_provider=rerank_provider,
                        llm_model=rerank_model
                    )
                    display_azure_openai_config(display_config, key_prefix="retrieve_result_azure")
            
            # Display retrieval status
            st.subheader("📊 Retrieval Status")
            status_col1, status_col2, status_col3, status_col4 = st.columns(4)
            with status_col1:
                st.metric("Chunks Retrieved", len(result.selected_chunks))
            with status_col2:
                candidates_count = len(result.trace.get("pinecone_candidates", [])) if hasattr(result, 'trace') and result.trace else 0
                st.metric("Pinecone Candidates", candidates_count)
            with status_col3:
                reranked_count = len(result.trace.get("rerank_results", [])) if hasattr(result, 'trace') and result.trace else 0
                st.metric("After Reranking", reranked_count)
            with status_col4:
                final_tokens = result.trace.get("final_token_count", 0) if hasattr(result, 'trace') and result.trace else 0
                st.metric("Total Tokens", f"{final_tokens:,}")
            
            # Pinecone Candidates Table (Before Reranking)
            st.subheader("📊 Pinecone Candidates (Before Reranking)")
            if result.trace and result.trace.get("pinecone_candidates"):
                candidates_df = []
                for idx, cand in enumerate(result.trace["pinecone_candidates"][:20], 1):
                    page_span = cand.get("page_span", [0, 0])
                    if isinstance(page_span, (list, tuple)) and len(page_span) >= 2:
                        page_span_str = f"{page_span[0]}-{page_span[1]}"
                    else:
                        page_span_str = str(page_span)
                    
                    candidates_df.append({
                        "Rank": idx,
                        "Chunk ID": cand.get("chunk_id", "unknown")[:20] + "...",
                        "Score": f"{cand.get('score', 0):.4f}",
                        "Doc ID": cand.get("doc_id", "unknown"),
                        "Page Span": page_span_str,
                        "Section Label": (cand.get("section_label", "") or "")[:50],
                        "Element Type": cand.get("element_type", "unknown")
                    })
                
                if candidates_df:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(candidates_df), use_container_width=True)
                else:
                    st.info("No candidates available")
            else:
                st.info("No Pinecone candidates in trace")
            
            # Reranked Results Table (After Reranking)
            st.subheader("🎯 Reranked Results (After Reranking)")
            if result.trace and result.trace.get("rerank_results"):
                reranked_df = []
                for idx, rerank in enumerate(result.trace["rerank_results"], 1):
                    score = rerank.get("relevance_score", 0)
                    # Color coding: 🟢 >= 70, 🟡 >= 40, 🔴 < 40
                    if score >= 70:
                        color = "🟢"
                    elif score >= 40:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    reranked_df.append({
                        "Rank": idx,
                        "Chunk ID": rerank.get("chunk_id", "unknown")[:20] + "...",
                        f"{color} Relevance": score,
                        "Answerability": "Yes" if score >= 40 else "No"
                    })
                
                if reranked_df:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(reranked_df), use_container_width=True)
                else:
                    st.info("No reranked results available")
            else:
                st.info("No reranked results in trace")
            
            # Display retrieved chunks with preview
            st.subheader("📝 Final Retrieved Chunks")
            
            # Show chunks in tabs
            result_tabs = st.tabs(["Chunks", "Context Pack", "Trace"])
            
            with result_tabs[0]:
                view_mode = st.radio("View Mode", ["text_for_embedding", "raw_md_fragment"], horizontal=True, key="retrieve_view_mode")
                
                for i, chunk in enumerate(result.selected_chunks, 1):
                    # Get relevance score if available
                    relevance_score = None
                    if result.trace and result.trace.get("rerank_results"):
                        for rerank in result.trace["rerank_results"]:
                            if rerank.get("chunk_id") == chunk.get("chunk_id"):
                                relevance_score = rerank.get("relevance_score", 0)
                                break
                    
                    title = f"Chunk {i}: {chunk.get('chunk_id', 'unknown')[:40]}..."
                    if relevance_score is not None:
                        title += f" (Relevance: {relevance_score})"
                    
                    chunk_id = chunk.get("chunk_id", "unknown")
                    with st.expander(title, expanded=(i <= 3)):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Metadata:**")
                            metadata = {
                                "chunk_id": chunk_id,
                                "doc_id": chunk.get("doc_id", "unknown"),
                                "element_type": chunk.get("element_type", "unknown"),
                                "page_nos": chunk.get("page_nos", []),
                                "section_label": chunk.get("section_label", ""),
                            }
                            if relevance_score is not None:
                                metadata["relevance_score"] = relevance_score
                                metadata["answerable"] = "Yes" if relevance_score >= 40 else "No"
                            if chunk.get("citation"):
                                metadata["citation"] = chunk.get("citation")
                            st.json(metadata)
                        with col2:
                            st.write("**Content:**")
                            content = chunk.get(view_mode, chunk.get("text_for_embedding", ""))
                            st.text_area("", content[:2000] + ("..." if len(content) > 2000 else ""), 
                                       height=300, key=f"retrieve_content_{i}_{chunk_id[:20]}", disabled=True)
            
            with result_tabs[1]:
                if hasattr(result, 'context_pack') and result.context_pack:
                    st.write("**Context Pack:**")
                    st.text_area("", result.context_pack, height=400, key="context_pack", disabled=True)
                    st.download_button(
                        "Download Context Pack",
                        result.context_pack,
                        file_name="context_pack.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("Context pack not available")
            
            with result_tabs[2]:
                if hasattr(result, 'trace') and result.trace:
                    st.json(result.trace)
                else:
                    st.info("Trace not available")
            
            # Download results
            st.subheader("📤 Export Results")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                results_json = json.dumps(result.selected_chunks, indent=2, default=str)
                st.download_button(
                    "Download Results (JSON)",
                    results_json,
                    file_name="retrieval_results.json",
                    mime="application/json",
                )
            with export_col2:
                if hasattr(result, 'context_pack') and result.context_pack:
                    st.download_button(
                        "Download Context Pack",
                        result.context_pack,
                        file_name="context_pack.txt",
                        mime="text/plain",
                    )
        except Exception as e:
            st.error(f"Error retrieving: {e}")
            logger.exception(e)
    
    # Show retrieval results if available (persists across tab switches)
    if st.session_state.get("retrieval_result"):
        result = st.session_state.retrieval_result
        st.divider()
        st.subheader("📊 Previous Retrieval Results")
        
        # Quick summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunks Retrieved", len(result.selected_chunks))
        with col2:
            candidates_count = len(result.trace.get("pinecone_candidates", [])) if hasattr(result, 'trace') and result.trace else 0
            st.metric("Pinecone Candidates", candidates_count)
        with col3:
            reranked_count = len(result.trace.get("rerank_results", [])) if hasattr(result, 'trace') and result.trace else 0
            st.metric("After Reranking", reranked_count)
        
        # Show Pinecone candidates table
        if result.trace and result.trace.get("pinecone_candidates"):
            with st.expander("📊 Pinecone Candidates (Before Reranking)", expanded=False):
                candidates_df = []
                for idx, cand in enumerate(result.trace["pinecone_candidates"][:20], 1):
                    page_span = cand.get("page_span", [0, 0])
                    if isinstance(page_span, (list, tuple)) and len(page_span) >= 2:
                        page_span_str = f"{page_span[0]}-{page_span[1]}"
                    else:
                        page_span_str = str(page_span)
                    
                    candidates_df.append({
                        "Rank": idx,
                        "Chunk ID": cand.get("chunk_id", "unknown")[:20] + "...",
                        "Score": f"{cand.get('score', 0):.4f}",
                        "Doc ID": cand.get("doc_id", "unknown"),
                        "Page Span": page_span_str,
                        "Section Label": (cand.get("section_label", "") or "")[:50],
                        "Element Type": cand.get("element_type", "unknown")
                    })
                
                if candidates_df:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(candidates_df), use_container_width=True)
        
        # Show reranked results table
        if result.trace and result.trace.get("rerank_results"):
            with st.expander("🎯 Reranked Results (After Reranking)", expanded=False):
                reranked_df = []
                for idx, rerank in enumerate(result.trace["rerank_results"], 1):
                    score = rerank.get("relevance_score", 0)
                    if score >= 70:
                        color = "🟢"
                    elif score >= 40:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    reranked_df.append({
                        "Rank": idx,
                        "Chunk ID": rerank.get("chunk_id", "unknown")[:20] + "...",
                        f"{color} Relevance": score,
                        "Answerability": "Yes" if score >= 40 else "No"
                    })
                
                if reranked_df:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(reranked_df), use_container_width=True)


def render_pipeline_tab():
    """Render end-to-end pipeline tab."""
    st.header("End-to-End Pipeline")
    st.info("This tab will run the complete pipeline: Parse → Chunk → Index → Retrieve")
    
    # Placeholder for full pipeline implementation
    st.write("Full pipeline implementation coming soon...")


if __name__ == "__main__":
    main()

