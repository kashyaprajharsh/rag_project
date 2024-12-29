import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import plotly.figure_factory as ff
from PIL import Image
import io
import time
from pdf import load_pdf_data, process_uploaded_files, split_text
from vectordb import (
    create_vectorstore,
    get_retriever,
    get_reranker_retriever,
    get_hybrid_retriever,
    get_hybrid_reranker_retriever,
    retrieve_documents,
    create_embedding_visualization,
    display_hyde_generations,
)
from rag import generate_all_rag_answers, RAG_TYPES
from agentic_rag import initialize_components, run_rag_bot_stream

from streamlit_option_menu import option_menu
from functools import partial
import matplotlib.pyplot as plt
from graphviz import Digraph
import logging
import sys
import validators
from typing import Any, List, Dict
from itertools import combinations
import difflib
import base64
import requests
import tempfile
import os


# Configure Streamlit Page
st.set_page_config(
    layout="wide",
    page_title="RAGExplorer",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# Setup Logging
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rag_explorer")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

logger = setup_logger()

# Initialize Session State with Default Values
def initialize_session_state() -> None:
    default_state = {
        "current_step": 0,
        "pdf_data": None,
        "pdf_loaded": False,
        "split_results": {},
        "vectorstore": None,
        "messages": [],
        "rag_results": [],
        "api_key": "",
        "recommended_splitter": None,
        "use_hyde": False,
        "retriever_created": False,
        "retrieval_done": False,
        "query": "",
        "current_retriever_type": "",
        # PDF Loading state
        "pdf_paths": [],
        "current_pdf_path": None,
        "pdf_source": "Upload PDFs",
        "uploaded_files": None,
        "selected_pdf_index": 0,
        # Text Splitting state
        "splitting_results": None,
        "previous_splitter_selection": ["Character"],  # Default selection
        "chunk_size": 1000,  # Default chunk size
        "chunk_overlap": 200,  # Default overlap
        "semantic_params": {
            "threshold_type": "percentile",
            "threshold_amount": 0.95,
            "number_of_chunks": 500
        },
        "agentic_rag_history": [],
        "agentic_rag_initialized": False,
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

logger.info("Starting application...")

# Add at the top of the file with other global variables
steps = ["Home", "PDF Loading", "Text Splitting", "Retriever", "RAG Chain", "Agentic RAG"]

# Centralized Error Handling
def handle_error(e: Exception, user_message: str = "An error occurred") -> None:
    st.error(f"{user_message}: {str(e)}")
    logger.error(f"{user_message}: {str(e)}")

# Validate PDF URLs
def validate_pdf_urls(urls: List[str]) -> List[str]:
    valid_urls = [url for url in urls if validators.url(url) and url.lower().endswith(".pdf")]
    return valid_urls

# Display PDF Content
def display_pdf_content(data: List[Any]) -> None:
    st.subheader("📄 PDF Content")
    st.info(f"Loaded {len(data)} pages.")

    current_pdf = ""
    for i, page in enumerate(data):
        if isinstance(page, dict):  # Header or footer
            if page["page_content"].startswith("--- Start of PDF:"):
                current_pdf = page["page_content"].split(":")[1].strip()[:-3]  # Extract PDF name
                st.markdown(f"### PDF: {current_pdf}")
            continue

        # Regular Document object
        with st.expander(f"Page {page.metadata['page']} of {current_pdf}"):
            st.text_area("Content", page.page_content, height=200, disabled=True)
            st.json(page.metadata)
            if "images" in page.metadata:
                for j, img_data in enumerate(page.metadata["images"]):
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, caption=f"Image {j+1}", use_container_width=True)

# Display Retrieved Documents
def display_retrieved_docs(docs: List[Any], retriever_name: str) -> None:
    for i, doc in enumerate(docs):
        with st.expander(f"{retriever_name} - Document {i+1}"):
            st.markdown(f"**Content:**\n{doc.page_content}")
            st.markdown("**Metadata:**")
            st.json(doc.metadata)

# Display RAG Results
def display_rag_results(results: Dict[str, Any], question_number: int) -> None:
    st.subheader(f"Results for Question {question_number}")
    rag_tabs = st.tabs(list(results.keys()))

    for rag_tab, (rag_type, result) in zip(rag_tabs, results.items()):
        with rag_tab:
            st.markdown("### Answer")
            st.write(result["output"])

            if st.button(
                f"Show Chain Visualization for {rag_type}",
                key=f"chain_viz_button_{question_number}_{rag_type}",
            ):
                visualize_rag_chain(result)

# Visualize RAG Chain
def visualize_rag_chain(chain_data: Dict[str, Any]) -> None:
    st.subheader("RunnableSequence")

    # Input and Output
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.text_area(
            "Input",
            value=chain_data["input"],
            height=100,
            disabled=True,
            key="chain_input",
        )
    with col2:
        st.subheader("Output")
        st.text_area(
            "Output",
            value=chain_data["output"],
            height=100,
            disabled=True,
            key="chain_output",
        )

    # Prompt Visualization
    with st.expander("Prompt"):
        if "prompt" in chain_data:
            st.code(chain_data["prompt"], language="text")
        else:
            st.info("Prompt information not available in the chain data.")

    # Metadata
    st.subheader("Metadata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Start Time",
            datetime.fromtimestamp(chain_data["startTime"] / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )
    with col2:
        st.metric(
            "End Time",
            datetime.fromtimestamp(chain_data["endTime"] / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )
    with col3:
        st.metric("Latency", f"{chain_data['latency']:.2f} s")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", chain_data["status"])
    with col2:
        st.metric("Total Tokens", chain_data["totalTokens"])
    with col3:
        st.metric("Cost", f"${chain_data['cost']:.6f}")

    # Step Durations
    with st.expander("Step Durations"):
        for step in chain_data["steps"]:
            st.text(f"{step['name']}: {step['duration']:.2f} s")

    # Context
    with st.expander("Context"):
        for i, doc in enumerate(chain_data["context"], 1):
            st.markdown(f"**Source {i}**")
            st.write(doc.page_content)
            st.json(doc.metadata)
            st.markdown("---")

# Display Splitting Results
def display_splitting_results(results: Dict[str, Any]) -> None:
    # Store results in session state
    st.session_state.splitting_results = results
    
    st.subheader("Splitter Metrics Comparison")
    metrics_df = pd.DataFrame({
        'Splitter': list(results.keys()),
        'Avg Chunk Size (chars)': [f"{r['avg_chunk_size']:.2f}" for r in results.values()],
        'Chunk Size Std Dev (chars)': [f"{np.sqrt(r['chunk_size_variance']):.2f}" for r in results.values()],
        'Coefficient of Variation': [f"{r['coefficient_of_variation']:.4f}" for r in results.values()],
        'Chunk Count': [r['chunk_count'] for r in results.values()],
        'Content Preservation (%)': [f"{r['content_preservation']*100:.2f}%" for r in results.values()],
        'Vocabulary Retention (%)': [f"{r['vocabulary_retention']*100:.2f}%" for r in results.values()],
        'Processing Time (s)': [f"{r['processing_time']:.3f}" for r in results.values()]
    })
    if any('semantic_coherence' in r for r in results.values()):
        metrics_df['Semantic Coherence'] = [
            f"{r.get('semantic_coherence', 'N/A'):.4f}" 
            if r.get('semantic_coherence') is not None else "N/A" 
            for r in results.values()
        ]

    st.dataframe(metrics_df)

    # Add a detailed explanation of each metric
    st.subheader("Metric Explanations")
    st.markdown("""
    - **Avg Chunk Size (chars)**: The average number of characters in each chunk.
    - **Chunk Size Std Dev (chars)**: The standard deviation of chunk sizes, indicating how much chunk sizes typically deviate from the average.
    - **Coefficient of Variation**: The ratio of the standard deviation to the mean, useful for comparing variability between splitters with different average chunk sizes. Lower values indicate more consistent chunk sizes.
    - **Chunk Count**: The total number of chunks produced by the splitter.
    - **Content Preservation (%)**: The percentage of original content (based on word frequency) preserved after splitting. Higher is better.
    - **Vocabulary Retention (%)**: The percentage of unique words from the original text retained in the split chunks. Higher is better.
    - **Processing Time (s)**: The time taken to perform the splitting operation, in seconds.
    - **Semantic Coherence**: (For semantic splitter only) A measure of how semantically related adjacent chunks are. Higher values indicate better semantic coherence.
    """)

    st.subheader("Chunk Size Distribution")
    hist_data = []
    group_labels = []
    for splitter, result in results.items():
        chunk_sizes = [len(chunk.page_content) for chunk in result['splits']]
        hist_data.append(chunk_sizes)
        group_labels.append(splitter)

    # Create a unique key based on the splitters being compared
    chart_key = f"chunk_size_dist_{'_'.join(sorted(results.keys()))}"
    
    fig = ff.create_distplot(hist_data, group_labels, bin_size=100)
    fig.update_layout(
        title_text='Distribution of Chunk Sizes',
        xaxis_title_text='Chunk Size (characters)',
        yaxis_title_text='Density'
    )
    st.plotly_chart(fig, key=chart_key)  # Use the unique key here

    st.subheader("Splitter Recommendation")
    recommendations = {
        "Consistency": min(results, key=lambda x: results[x]['coefficient_of_variation']),
        "Content Preservation": max(results, key=lambda x: results[x]['content_preservation']),
        "Processing Speed": min(results, key=lambda x: results[x]['processing_time'])
    }
    if any('semantic_coherence' in r for r in results.values()):
        recommendations["Semantic Coherence"] = max(
            (x for x in results if results[x].get('semantic_coherence') is not None), 
            key=lambda x: results[x]['semantic_coherence'],
            default=None
        )

    for metric, recommended_splitter in recommendations.items():
        if recommended_splitter is not None:
            st.success(
                f"Based on {metric}, the recommended splitter is: **{recommended_splitter}**"
            )


# Define Pages
def home_page() -> None:
    # Create two columns for the entire content including header
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.write("")
        st.header("Welcome to RAGExplorer")
        st.write(
            "This application helps you understand Retrieval-Augmented Generation (RAG) and its internal workings."
        )
        # What is RAG section
        st.subheader("What is RAG?")
        st.write(
            """
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with 
            external knowledge retrieval. It allows AI models to access and utilize information beyond their training data, 
            leading to more accurate and up-to-date responses.
            """
        )

        # Add OpenAI dependency warning for Agentic RAG
        st.info(
            """
            📝 **Note**: The Agentic RAG feature currently only works with OpenAI models. 
            Support for other LLM providers will be added in future updates.
            """
        )

        # How to use section
        st.subheader("How to use this app")
        st.write(
            """
            1. **Load PDFs**: Start by uploading your PDF documents.
            2. **Text Splitting**: Experiment with different text splitting techniques.
            3. **Retriever Methods**: Explore various retriever options.
            4. **RAG Chain**: Test the RAG chain with your own queries.
            """
        )

    with right_col:
        # Reduce vertical spacing before the image
        st.write("")  # Reduced from two write() statements to one
        # Add AI image with curved edges using HTML/CSS
        ai_image = Image.open("AI_image.png")
        # Convert the image to bytes
        img_bytes = io.BytesIO()
        ai_image.save(img_bytes, format='PNG')
        img_str = base64.b64encode(img_bytes.getvalue()).decode()
        
        # HTML with CSS for curved edges and adjusted margin
        st.markdown(
            f"""
            <style>
            .curved-image {{
                border-radius: 25px;
                overflow: hidden;
                margin-top: -100px;  /* Added negative margin to move image up */
            }}
            .curved-image img {{
                width: 100%;
                height: auto;
                display: inline;
            }}
            </style>
            <div class="curved-image">
                <img src="data:image/png;base64,{img_str}">
            </div>
            """,
            unsafe_allow_html=True
        )

# Add this function to handle PDF display for different sources
def get_pdf_display_path(pdf_source):
    """
    Returns a local file path for the PDF, downloading if necessary.
    """
    if isinstance(pdf_source, (str, bytes)):
        if isinstance(pdf_source, str) and pdf_source.startswith('http'):
            # Download from URL
            try:
                response = requests.get(pdf_source)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    return tmp_file.name
            except Exception as e:
                st.error(f"Failed to download PDF: {str(e)}")
                return None
        else:
            # Local file path
            return pdf_source
    else:
        # Uploaded file
        try:
            # Read the content of the uploaded file
            pdf_content = pdf_source.read()
            # Create a temporary file with the content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                return tmp_file.name
        except Exception as e:
            st.error(f"Failed to save uploaded PDF: {str(e)}")
            return None

def displayPDF(file_path):
    """
    Display PDF in the Streamlit app using PDF.js
    """
    try:
        # Read PDF file
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # PDF.js viewer HTML
        pdf_display = f'''
            <iframe
                src="https://mozilla.github.io/pdf.js/web/viewer.html?file=data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="600px"
                style="border: none;"
            ></iframe>
        '''
        
        # Display PDF
        st.markdown(pdf_display, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        try:
            # Provide download option as fallback
            with open(file_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f,
                    file_name="document.pdf",
                    mime="application/pdf"
                )
        except Exception as download_error:
            st.error(f"Error creating download button: {str(download_error)}")

def pdf_loading_page() -> None:
    st.header("📄 PDF Loading")

    main_col, viewer_col = st.columns([3, 2])

    with main_col:
        # Use session state for pdf_source
        st.session_state.pdf_source = st.radio(
            "Choose PDF source:",
            ["Upload PDFs", "Use provided PDF", "Use built-in link", "Use custom URLs"],
            horizontal=True,
            key="pdf_source_radio",
            index=["Upload PDFs", "Use provided PDF", "Use built-in link", "Use custom URLs"].index(st.session_state.pdf_source)
        )

        pdf_paths = []
        current_pdf_path = None

        if st.session_state.pdf_source == "Upload PDFs":
            # Use session state for uploaded files
            uploaded_files = st.file_uploader(
                "Upload PDF files", 
                type="pdf", 
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            if uploaded_files:
                # Store the uploaded files in session state
                st.session_state.uploaded_files = uploaded_files
                pdf_paths = process_uploaded_files(uploaded_files)
                if pdf_paths:
                    # Seek to beginning of file before reading
                    uploaded_files[st.session_state.selected_pdf_index].seek(0)
                    current_pdf_path = get_pdf_display_path(uploaded_files[st.session_state.selected_pdf_index])
                st.success(f"{len(uploaded_files)} file(s) uploaded")
                
                if len(uploaded_files) > 1:
                    selected_index = st.selectbox(
                        "Select PDF to view",
                        range(len(uploaded_files)),
                        format_func=lambda x: uploaded_files[x].name,
                        key="pdf_selector",
                        index=st.session_state.selected_pdf_index
                    )
                    st.session_state.selected_pdf_index = selected_index
                    # Seek to beginning of file before reading
                    uploaded_files[selected_index].seek(0)
                    current_pdf_path = get_pdf_display_path(uploaded_files[selected_index])
            # Restore previously uploaded files if they exist
            elif st.session_state.get("uploaded_files"):
                uploaded_files = st.session_state.uploaded_files
                pdf_paths = process_uploaded_files(uploaded_files)
                if pdf_paths:
                    # Seek to beginning of file before reading
                    uploaded_files[st.session_state.selected_pdf_index].seek(0)
                    current_pdf_path = get_pdf_display_path(uploaded_files[st.session_state.selected_pdf_index])
                st.success(f"{len(uploaded_files)} file(s) previously uploaded")
                
                if len(uploaded_files) > 1:
                    selected_index = st.selectbox(
                        "Select PDF to view",
                        range(len(uploaded_files)),
                        format_func=lambda x: uploaded_files[x].name,
                        key="pdf_selector",
                        index=st.session_state.selected_pdf_index
                    )
                    st.session_state.selected_pdf_index = selected_index
                    # Seek to beginning of file before reading
                    uploaded_files[selected_index].seek(0)
                    current_pdf_path = get_pdf_display_path(uploaded_files[selected_index])

        elif st.session_state.pdf_source == "Use provided PDF":
            pdf_paths = ["LargeLanguageModel.pdf"]
            current_pdf_path = pdf_paths[0]
            st.info(f"Using provided PDF: {pdf_paths[0]}")

        elif st.session_state.pdf_source == "Use built-in link":
            pdf_paths = ["https://arxiv.org/pdf/2410.07176.pdf"]
            current_pdf_path = get_pdf_display_path(pdf_paths[0])
            st.info(f"Using built-in link: {pdf_paths[0]}")

        else:  # Use custom URLs
            # Use session state for URLs
            raw_urls = st.text_area(
                "Enter PDF URLs (one per line):",
                value="\n".join(st.session_state.get("custom_urls", [])),
                key="url_input"
            ).split("\n")
            
            pdf_paths = [url.strip() for url in raw_urls if url.strip()]
            if pdf_paths:
                st.session_state.custom_urls = pdf_paths
                valid_pdf_paths = validate_pdf_urls(pdf_paths)
                if len(valid_pdf_paths) < len(pdf_paths):
                    st.warning("Some URLs may not be valid PDFs and have been excluded.")
                pdf_paths = valid_pdf_paths
                if pdf_paths:
                    current_pdf_path = get_pdf_display_path(pdf_paths[st.session_state.selected_pdf_index])
                    
                    if len(pdf_paths) > 1:
                        selected_index = st.selectbox(
                            "Select PDF to view",
                            range(len(pdf_paths)),
                            format_func=lambda x: pdf_paths[x],
                            key="url_selector",
                            index=st.session_state.selected_pdf_index
                        )
                        st.session_state.selected_pdf_index = selected_index
                        current_pdf_path = get_pdf_display_path(pdf_paths[selected_index])
            else:
                st.info("Please enter valid PDF URLs ending with '.pdf'")

        # Store paths and current path in session state
        st.session_state.pdf_paths = pdf_paths
        st.session_state.current_pdf_path = current_pdf_path

        with st.expander("Advanced Options"):
            loading_method = st.selectbox(
                "Loading method:",
                ["Normal", "With Images"],
                help="Choose 'With Images' to extract images from PDFs",
            )

        if st.button("📥 Load PDFs", key="load_pdfs_button"):
            if pdf_paths:
                with st.spinner("Loading PDFs..."):
                    try:
                        data = load_pdf_data(
                            pdf_paths, method=loading_method.lower().replace(" ", "_")
                        )
                        st.session_state.pdf_data = data
                        st.session_state.pdf_loaded = True
                        st.success(f"✅ {len(pdf_paths)} PDF(s) loaded successfully!")

                        # Display PDF content
                        display_pdf_content(data)
                    except Exception as e:
                        handle_error(e, "Error loading PDFs")
            else:
                st.error("❗ Please provide valid PDF source(s).")

    # Display PDF viewer in the side column
    with viewer_col:
        st.markdown("### PDF Preview")
        if st.session_state.current_pdf_path:
            displayPDF(st.session_state.current_pdf_path)
        else:
            st.info("Select a PDF to preview")

    # Clean up temporary files when the session ends
    def cleanup_temp_files():
        if current_pdf_path and current_pdf_path.startswith(tempfile.gettempdir()):
            try:
                os.remove(current_pdf_path)
            except:
                pass
    
    st.session_state['_cleanup_temp_files'] = cleanup_temp_files

def text_splitting_page() -> None:
    st.header("✂️ Text Splitting - Chunking")

    if not st.session_state.get("pdf_loaded", False):
        st.warning("Please load a PDF first in the PDF Loading step.")
        return

    st.info(
        """
        Text splitting (chunking) is crucial for processing large documents. It breaks the text into smaller, manageable chunks.
        Different splitters work better for different types of documents. Experiment to find the best one for your needs.
        """
    )

    splitter_types = {
        "Character": "Splits text based on a fixed number of characters. Simple but may break words.",
        "Recursive": "Intelligently splits text into chunks, trying to keep sentences and paragraphs intact.",
        "Token": "Splits text based on the number of tokens (words or subwords). Useful for maintaining context.",
        "Semantic": "Uses embeddings to split text based on semantic meaning. Requires OpenAI API key.",
    }

    # Use session state for splitter selection with previous value
    selected_splitters = st.multiselect(
        "Choose splitter type(s):",
        list(splitter_types.keys()),
        default=st.session_state.previous_splitter_selection,
        help="Select one or more splitters to compare different splitting techniques.",
        key="splitter_selection"
    )

    # Check if selection has changed
    selection_changed = selected_splitters != st.session_state.previous_splitter_selection
    st.session_state.previous_splitter_selection = selected_splitters.copy()

    # Clear previous results if selection changed
    if selection_changed:
        st.session_state.splitting_results = None

    for splitter, description in splitter_types.items():
        if splitter in selected_splitters:
            st.markdown(f"**{splitter} Splitter**: {description}")

    col1, col2 = st.columns(2)
    with col1:
        # Use session state for chunk size
        chunk_size = st.slider(
            "Chunk size",
            100,
            2000,
            st.session_state.chunk_size,
            help="Number of characters/tokens per chunk. Larger sizes retain more context but may be less manageable.",
            key="chunk_size_slider"
        )
        st.session_state.chunk_size = chunk_size
    
    with col2:
        # Use session state for chunk overlap
        chunk_overlap = st.slider(
            "Chunk overlap",
            0,
            500,
            st.session_state.chunk_overlap,
            help="Number of overlapping characters/tokens between chunks to preserve context across splits.",
            key="chunk_overlap_slider"
        )
        st.session_state.chunk_overlap = chunk_overlap

    semantic_params = st.session_state.semantic_params.copy()
    if "Semantic" in selected_splitters:
        st.warning("⚠️ The Semantic Splitter may take a considerable amount of time for large PDFs. Please be patient.")
        if not st.session_state.api_key:
            st.error("❗ OpenAI API Key is required for the Semantic Splitter. Please provide it in the sidebar.")
            return
        with st.expander("Semantic Splitter Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                semantic_params["threshold_type"] = st.selectbox(
                    "Breakpoint Threshold Type",
                    ["percentile", "standard_deviation", "interquartile"],
                    index=["percentile", "standard_deviation", "interquartile"].index(
                        st.session_state.semantic_params["threshold_type"]
                    ),
                    help="Method to determine breakpoints in the text",
                    key="threshold_type_select"
                )
            with col2:
                semantic_params["threshold_amount"] = st.slider(
                    "Breakpoint Threshold Amount",
                    0.0,
                    1.0,
                    st.session_state.semantic_params["threshold_amount"],
                    0.01,
                    help="Threshold for determining breakpoints",
                    key="threshold_amount_slider"
                )
            semantic_params["number_of_chunks"] = st.slider(
                "Number of Chunks",
                10,
                1000,
                st.session_state.semantic_params["number_of_chunks"],
                help="Target number of chunks to create",
                key="number_of_chunks_slider"
            )
            st.session_state.semantic_params = semantic_params

    # Show Process PDF button only if no results exist for current selection
    if not st.session_state.get('splitting_results') or selection_changed:
        if st.button("Process PDF", key="process_pdf_button"):
            if "Semantic" in selected_splitters and not st.session_state.api_key:
                st.error("❗ OpenAI API Key is required for the Semantic Splitter.")
                return
            if selected_splitters:
                with st.spinner("Processing..."):
                    try:
                        results = {}
                        for splitter in selected_splitters:
                            if splitter.lower() == "semantic":
                                results[splitter] = split_text(
                                    st.session_state.pdf_data,
                                    splitter.lower(),
                                    semantic_params=semantic_params,
                                    api_key=st.session_state.get("api_key"),
                                )
                            else:
                                results[splitter] = split_text(
                                    st.session_state.pdf_data,
                                    splitter.lower(),
                                    chunk_size,
                                    chunk_overlap,
                                    api_key=st.session_state.get("api_key"),
                                )

                            st.success(
                                f"{splitter} splitter: Text split into {results[splitter]['chunk_count']} chunks in {results[splitter]['processing_time']:.2f} seconds"
                            )

                        display_splitting_results(results)
                        st.session_state.split_results = {
                            splitter: result["splits"] for splitter, result in results.items()
                        }
                        st.session_state.splitting_results = results
                        st.session_state.recommended_splitter = min(
                            results, key=lambda x: results[x]["coefficient_of_variation"]
                        )

                    except Exception as e:
                        handle_error(e, "Error processing PDF")
            else:
                st.error("Please select at least one splitter type.")
    
    # Display existing results if available
    elif st.session_state.get('splitting_results'):
        display_splitting_results(st.session_state.splitting_results)


# Page: Retriever

def retriever_page() -> None:
    st.header("🔍 Vector Store and Retriever")

    if "pdf_data" not in st.session_state or not st.session_state.pdf_data:
        st.warning("Please load a PDF first in the PDF Loading step.")
        st.stop()

    if "split_results" not in st.session_state or not st.session_state.split_results:
        st.warning("Please process the PDF in the Text Splitting step first.")
        st.stop()

    if "recommended_splitter" not in st.session_state:
        st.warning("No recommended splitter found. Please complete the Text Splitting step.")
        st.stop()

    st.subheader("Text Splitting")
    selected_splitter = st.selectbox(
        "Choose splitter to use:",
        list(st.session_state.split_results.keys()),
        index=list(st.session_state.split_results.keys()).index(
            st.session_state.recommended_splitter
        ),
        help="Select the text splitter to use for creating the vector store",
    )

    st.info(
        f"Using {len(st.session_state.split_results[selected_splitter])} chunks from the {selected_splitter} splitter."
    )

    st.subheader("Vector Store")
    use_hyde = st.checkbox(
        "Use HYDE (Hypothetical Document Embedder)", value=False
    )
    batch_size = st.number_input(
        "Batch size", min_value=100, max_value=10000, value=5000, step=100
    )

    if st.button("Create Vector Store", key="create_vector_store_button"):
        if not st.session_state.api_key:
            st.error("❗ OpenAI API Key is required to create the Vector Store.")
            return
        if selected_splitter not in st.session_state.split_results:
            st.error(f"❗ Splitter '{selected_splitter}' not found in split results.")
            return
        with st.spinner("Creating Vector Store..."):
            try:
                splits = st.session_state.split_results[
                    st.session_state.recommended_splitter
                ]
                vectorstore = create_vectorstore(
                    splits=splits,
                    batch_size=batch_size,
                    use_hyde=use_hyde,
                    api_key=st.session_state.get("api_key"),
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.use_hyde = use_hyde
                st.session_state.vector_store_created = True  # Track vector store creation
                st.success(
                    f"{'HYDE ' if use_hyde else ''}Vector Store created successfully!"
                )
            except Exception as e:
                handle_error(e, "Error creating vector store")

    # Show retriever settings only if vector store is created
    if st.session_state.get("vector_store_created", False):
        st.subheader("Retriever Settings")
        retriever_type = st.selectbox(
            "Retriever type",
            [
                "Vector Store Retriever",
                "Vector Store Retriever with Reranker",
                "Ensemble Retriever (BM25 + Vector Store)",
                "Ensemble Retriever with Reranker"
            ],
            key="retriever_type",
        )

        search_type = st.selectbox(
            "Search type", ["similarity", "mmr"], key="retriever_search_type"
        )
        k = st.slider(
            "Number of documents to retrieve (k)", 1, 20, 5, key="retriever_k"
        )
        if search_type == "mmr":
            fetch_k = st.slider(
                "Number of documents to fetch for MMR (fetch_k)",
                k,
                50,
                20,
                key="retriever_fetch_k",
            )
        else:
            fetch_k = k

        if "Reranker" in retriever_type:
            reranker_type = st.selectbox(
                "Reranker type", 
                ["cohere", "huggingface (BAAI/bge-reranker-base)"], 
                key="reranker_type"
            )
            top_n = st.slider(
                "Number of documents to rerank (top_n)", 1, 10, 3, key="reranker_top_n"
            )

        if "Ensemble" in retriever_type:
            bm25_weight = st.slider(
                "BM25 Weight",
                0.0,
                1.0,
                0.5,
                0.1,
                key="bm25_weight",
                help="Weight for BM25 retrieval in the ensemble retriever.",
            )
            vector_weight = 1 - bm25_weight
            st.info(f"Vector Store Weight: {vector_weight:.1f}")

        if st.button("Create Retriever", key="create_retriever_button"):
            with st.spinner("Creating Retriever..."):
                try:
                    base_retriever = get_retriever(
                        st.session_state.vectorstore,
                        search_type,
                        k,
                        fetch_k,
                    )
                    st.session_state.base_retriever = base_retriever

                    if "Reranker" in retriever_type:
                        reranker_retriever = get_reranker_retriever(
                            base_retriever, reranker_type, top_n
                        )
                        st.session_state.reranker_retriever = reranker_retriever

                    if "Ensemble" in retriever_type:
                        hybrid_retriever = get_hybrid_retriever(
                            st.session_state.split_results[
                                st.session_state.recommended_splitter
                            ],
                            base_retriever,
                            bm25_weight,
                            vector_weight,
                        )
                        st.session_state.hybrid_retriever = hybrid_retriever

                    if retriever_type == "Ensemble Retriever with Reranker":
                        hybrid_reranker_retriever = get_hybrid_reranker_retriever(
                            st.session_state.split_results[
                                st.session_state.recommended_splitter
                            ],
                            base_retriever,
                            bm25_weight,
                            vector_weight,
                            reranker_type,
                            top_n,
                        )
                        st.session_state.hybrid_reranker_retriever = hybrid_reranker_retriever

                    # Store the selected retriever type and its corresponding retriever
                    st.session_state.current_retriever_type = retriever_type
                    if retriever_type == "Vector Store Retriever":
                        st.session_state.selected_retriever = base_retriever
                    elif retriever_type == "Vector Store Retriever with Reranker":
                        st.session_state.selected_retriever = reranker_retriever
                    elif retriever_type == "Ensemble Retriever (BM25 + Vector Store)":
                        st.session_state.selected_retriever = hybrid_retriever
                    else:  # Ensemble Retriever with Reranker
                        st.session_state.selected_retriever = hybrid_reranker_retriever
                    st.session_state.retriever_created = True  # Track retriever creation
                    st.success(
                        f"{'HYDE ' if st.session_state.use_hyde else ''}Vector Store and {retriever_type} created successfully!"
                    )
                    st.session_state.retriever_created = True  # Track retriever creation
                except Exception as e:
                    handle_error(e, "Error creating retriever")

    # Show test options if retriever is created
    if st.session_state.get("retriever_created", False):
        st.subheader("Test Retriever")
        
        # Store query and retrieved documents in session state
        if "current_query" not in st.session_state:
            st.session_state.current_query = ""
        if "current_retrieved_docs" not in st.session_state:
            st.session_state.current_retrieved_docs = {}
        if "hyde_doc" not in st.session_state:
            st.session_state.hyde_doc = None
            
        query = st.text_input(
            "Enter a query to test the retriever:", 
            key="retriever_query"
        )
        
        # Modified retrieval logic to handle same query and show HYDE docs
        if st.button("Retrieve", key="retrieve_button"):
            if not query.strip():
                st.error("❗ Please enter a valid query.")
            else:
                with st.spinner("Retrieving documents..."):
                    try:
                        # If HYDE was used, generate and store the document
                        if st.session_state.get("use_hyde", False):
                            hyde_doc = display_hyde_generations(query, return_doc=True)
                            st.session_state.hyde_doc = hyde_doc
                        
                        # Store retrieved documents in session state
                        base_docs = retrieve_documents(
                            st.session_state.base_retriever, query
                        )
                        st.session_state.current_retrieved_docs["base"] = base_docs

                        if st.session_state.current_retriever_type != "Vector Store Retriever":
                            if st.session_state.current_retriever_type == "Vector Store Retriever with Reranker":
                                comparison_docs = retrieve_documents(
                                    st.session_state.reranker_retriever, query
                                )
                            elif st.session_state.current_retriever_type == "Ensemble Retriever (BM25 + Vector Store)":
                                comparison_docs = retrieve_documents(
                                    st.session_state.hybrid_retriever, query
                                )
                            elif st.session_state.current_retriever_type == "Ensemble Retriever with Reranker":
                                comparison_docs = retrieve_documents(
                                    st.session_state.hybrid_reranker_retriever, query
                                )
                            st.session_state.current_retrieved_docs["comparison"] = comparison_docs
                        
                        st.session_state.retrieval_done = True
                        st.session_state.current_query = query
                    except Exception as e:
                        handle_error(e, "Error retrieving documents")

        # Display retrieved documents if they exist
        if st.session_state.get("retrieval_done", False) and st.session_state.get("current_retrieved_docs"):
            st.subheader("Retrieved Documents")

            # Display HYDE document if it exists (in an expander)
            if st.session_state.get("use_hyde", False) and st.session_state.get("hyde_doc"):
                with st.expander("HYDE Generated Document", expanded=True):
                    st.markdown(f"**Generated Hypothetical Document:**\n\n{st.session_state.hyde_doc}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Base Retriever")
                if "base" in st.session_state.current_retrieved_docs:
                    display_retrieved_docs(
                        st.session_state.current_retrieved_docs["base"], 
                        "Base Retriever"
                    )

            with col2:
                if st.session_state.current_retriever_type != "Vector Store Retriever":
                    st.markdown(f"### {st.session_state.current_retriever_type}")
                    if "comparison" in st.session_state.current_retrieved_docs:
                        display_retrieved_docs(
                            st.session_state.current_retrieved_docs["comparison"],
                            st.session_state.current_retriever_type
                        )
                else:
                    st.markdown("### Comparison")
                    st.info(
                        "No comparison retriever selected. Using Vector Store Retriever only."
                    )

            # Embedding visualization section
            st.subheader("Embedding Visualization")
            method = st.selectbox(
                "Choose visualization method", 
                ["pacmap", "umap", "tsne"]
            )
            dimensions = st.radio("Choose dimensions", [2, 3])
            sample_size = st.number_input(
                "Sample size (leave 0 for all)",
                min_value=0,
                max_value=10000,
                value=0,
                step=100,
            )

            if st.button("Show Embedding Visualization", key="show_embedding_viz_button"):
                with st.spinner("Creating embedding visualization..."):
                    try:
                        retrieved_docs = {
                            "Base": st.session_state.current_retrieved_docs["base"]
                        }
                        if "comparison" in st.session_state.current_retrieved_docs:
                            retrieved_docs[st.session_state.current_retriever_type] = (
                                st.session_state.current_retrieved_docs["comparison"]
                            )

                        fig = create_embedding_visualization(
                            st.session_state.vectorstore,
                            st.session_state.split_results[
                                st.session_state.recommended_splitter
                            ],
                            st.session_state.current_query,
                            retrieved_docs,
                            method=method,
                            sample_size=sample_size if sample_size > 0 else None,
                            dimensions=dimensions,
                            api_key=st.session_state.get("api_key"),
                        )
                        st.plotly_chart(fig)
                    except Exception as e:
                        handle_error(e, "Error creating embedding visualization")


def rag_chain_page() -> None:
    st.header("🔗 RAG Chain")

    if "pdf_data" not in st.session_state or not st.session_state.pdf_data:
        st.warning("Please load a PDF first in the PDF Loading step.")
        st.stop()

    if "vectorstore" not in st.session_state or not st.session_state.vectorstore:
        st.warning("Please create a Vector Store in the Retriever step first.")
        st.stop()

    if not st.session_state.get("api_key"):
        st.warning("Please provide your OpenAI API Key in the sidebar to use the RAG Chain.")
        st.stop()

    # Display chat messages and results from history
    for i in range(0, len(st.session_state.messages), 2):
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[i]["content"])

        if i + 1 < len(st.session_state.messages):
            with st.chat_message("assistant"):
                st.markdown(st.session_state.messages[i + 1]["content"])

            if i // 2 < len(st.session_state.rag_results):
                display_rag_results(st.session_state.rag_results[i // 2], i // 2 + 1)

    # Chat input
    prompt = st.chat_input("What is your question?", key="rag_chat_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating answers..."):
            try:
                retrievers = {
                    "base_retriever": st.session_state.get("base_retriever"),
                    "reranker_retriever": st.session_state.get("reranker_retriever"),
                    "hybrid_retriever": st.session_state.get("hybrid_retriever"),
                    "hybrid_reranker_retriever": st.session_state.get("hybrid_reranker_retriever"),
                }

                results = generate_all_rag_answers(
                    prompt,
                    retrievers,
                    st.session_state.get("api_key"),
                )

                if results:
                    st.session_state.rag_results.append(results)

                    used_retrievers = list(results.keys())
                    missing_retrievers = [r for r in RAG_TYPES if r not in used_retrievers]

                    response = "I've generated answers using the following RAG methods: "
                    response += ", ".join(used_retrievers) + ". "

                    if missing_retrievers:
                        response += (
                            f"Note that the following retrievers were not available: {', '.join(missing_retrievers)}. "
                            "Please create these retrievers in the Retriever step if you want to use them."
                        )

                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(
                        "No results were generated. Please check if any retrievers are available and properly configured."
                    )
            except Exception as e:
                handle_error(e, "Error generating answers")

        st.rerun()

# Add a new page for Agentic RAG
def agentic_rag_page() -> None:
    st.header("🤖 Agentic RAG")

    # Check prerequisites
    if "pdf_data" not in st.session_state or not st.session_state.pdf_data:
        st.warning("Please load a PDF first in the PDF Loading step.")
        return

    if "vectorstore" not in st.session_state or not st.session_state.vectorstore:
        st.warning("Please create a Vector Store in the Retriever step first.")
        return
        
    if not st.session_state.get("selected_retriever"):
        st.warning("Please select a retriever in the Retriever step first.")
        return

    if not st.session_state.get("api_key"):
        st.warning("Please provide your API Key in the sidebar to use Agentic RAG.")
        return

    # Check if using OpenAI
    if st.session_state.get("llm_provider") != "OpenAI":
        st.error(
            """
            ⚠️ Agentic RAG currently only works with OpenAI models. 
            Please switch to OpenAI in the sidebar to use this feature.
            Support for other LLM providers will be added in future updates.
            """
        )
        return

    # Add workflow diagram
    st.markdown("""
    Below is the workflow diagram showing how the Agentic RAG system processes queries:
    """)
    
    # Display the workflow graph directly
    st.image("workflow_graph.png",width=225, caption="Agentic RAG Workflow Diagram")

    st.markdown("""
    ### About Agentic RAG
    This component uses an intelligent agent to:
    - Route questions to the most appropriate source (vectorstore or web search)
    - Grade document relevance
    - Check for hallucinations
    - Transform queries for better retrieval
    """)

    # Initialize components if not already done
    if not st.session_state.get("agentic_rag_initialized"):
        try:
            initialize_components(api_key=st.session_state.get("api_key"))
            st.session_state.agentic_rag_initialized = True
        except Exception as e:
            st.error(f"Error initializing Agentic RAG components: {str(e)}")
            return

    # Initialize chat history if it doesn't exist
    if "agentic_messages" not in st.session_state:
        st.session_state.agentic_messages = []

    # Display chat history
    for message in st.session_state.agentic_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask a question about your documents...", key="agentic_chat_input"):
        # Add user message to chat history
        st.session_state.agentic_messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Process query and display assistant response
        with st.chat_message("assistant"):
            try:
                # Create two columns for the response
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    answer_container = st.empty()
                    
                with col2:
                    with st.expander("Process Details", expanded=True):
                        process_container = st.empty()
                
                # Track the final answer and process details
                final_answer = ""
                process_details = []
                
                # Process the query and update the UI in real-time
                for output in run_rag_bot_stream(query, api_key=st.session_state.get("api_key")):
                    if output["type"] == "generation":
                        final_answer = output["content"]
                        with answer_container:
                            st.markdown("### Answer")
                            st.markdown(final_answer)
                    
                    # Update process details
                    process_details.append(f"**{output['key']}**:")
                    process_details.append(f"{str(output['content'])}\n")
                    with process_container:
                        st.markdown("\n".join(process_details))
                
                # Add the final answer to chat history
                if final_answer:
                    st.session_state.agentic_messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })
                else:
                    error_msg = "No answer was generated. Please try rephrasing your question."
                    st.error(error_msg)
                    st.session_state.agentic_messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })
                    
            except Exception as e:
                error_message = f"Error running Agentic RAG: {str(e)}"
                st.error(error_message)
                logger.error(f"Agentic RAG error: {str(e)}", exc_info=True)
                st.session_state.agentic_messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_message}"
                })

# Main Function
def main() -> None:
    st.title("RAGExplorer")
    st.markdown("Explore the inner workings of Retrieval-Augmented Generation (RAG)")

    # Add OpenAI API Key input in the sidebar
    with st.sidebar:
        # Add RAG logo at the top of sidebar
        rag_logo = Image.open("rag.png")
        st.image(rag_logo, width=220)
        
        st.markdown("## LLM Selection")
        # Default to Gemini
        llm_provider = st.selectbox(
            "Choose LLM Provider:",
            ["Google Gemini", "OpenAI"],
            key="llm_provider"
        )
        
        if llm_provider == "OpenAI":
            st.markdown("## OpenAI API Key")
            api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="openai_api_key_input")
            if api_key:
                st.success("OpenAI API Key provided!")
                st.session_state.api_key = api_key
            else:
                st.warning("Please enter your OpenAI API Key to use OpenAI models.")
        else:  # Google Gemini
            st.success("Using Google Gemini - No API key required!")
            st.session_state.api_key = st.secrets.get("GOOGLE_API_KEY")
            
        st.markdown("---")
        # Determine which steps are available
        steps_available = {
            "Home": True,
            "PDF Loading": True,
            "Text Splitting": "pdf_data" in st.session_state and st.session_state.pdf_data,
            "Retriever": "split_results" in st.session_state and st.session_state.split_results,
            "RAG Chain": "vectorstore" in st.session_state and st.session_state.vectorstore and st.session_state.get("api_key"),
            "Agentic RAG": "vectorstore" in st.session_state and st.session_state.vectorstore and st.session_state.get("api_key")
        }

        # Sidebar Navigation using option_menu
        steps = ["Home", "PDF Loading", "Text Splitting", "Retriever", "RAG Chain", "Agentic RAG"]
        selected = option_menu(
            menu_title=None,
            options=steps,
            icons=["house", "file-pdf", "scissors", "search", "link", "robot"],
            menu_icon="cast",
            default_index=st.session_state.current_step,
        )

        # Update current_step based on sidebar selection, only if the step is available
        if steps_available[selected]:
            st.session_state.current_step = steps.index(selected)
        else:
            st.error(f"Please complete previous steps before accessing {selected}.")

        # Progress Indicator based on selected page
        current_step = st.session_state.current_step
        st.progress((current_step + 1) / len(steps), f"Step {current_step + 1} of {len(steps)}")

        # About Section
        with st.expander("About RAGExplorer"):
            st.write(
                "RAGExplorer helps you understand the inner workings of Retrieval-Augmented Generation."
            )
            st.write(
                "Navigate through the steps to explore each component of the RAG process."
            )

    # Define pages
    pages = {
        "Home": home_page,
        "PDF Loading": pdf_loading_page,
        "Text Splitting": text_splitting_page,
        "Retriever": retriever_page,
        "RAG Chain": rag_chain_page,
        "Agentic RAG": agentic_rag_page,
    }

    # Display the selected page
    pages[steps[st.session_state.current_step]]()

    # Navigation Buttons
    st.markdown("---")
    col_prev, col_center, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.session_state.current_step > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_step -= 1
                st.rerun()

    with col_next:
        if st.session_state.current_step < len(steps) - 1:
            next_step = steps[st.session_state.current_step + 1]
            # Determine if the current step is completed
            step_completed = False
            current_step_name = steps[st.session_state.current_step]

            if current_step_name == "Home":
                step_completed = True  # Home is always completed
            elif current_step_name == "PDF Loading":
                step_completed = st.session_state.get("pdf_loaded", False)
            elif current_step_name == "Text Splitting":
                step_completed = bool(st.session_state.get("split_results"))
            elif current_step_name == "Retriever":
                step_completed = st.session_state.get("retriever_created", False)
            elif current_step_name == "RAG Chain":
                step_completed = st.session_state.get("vectorstore", False) and st.session_state.get("api_key", False)

            if st.button("Next ➡️"):
                if step_completed:
                    st.session_state.current_step += 1
                    st.rerun()
                else:
                    st.error(f"Please complete the **{current_step_name}** step before proceeding to **{next_step}**.")

if __name__ == "__main__":
    initialize_session_state()
    main()

