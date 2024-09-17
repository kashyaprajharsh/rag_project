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
)
from rag import generate_all_rag_answers, RAG_TYPES
from visual import visualize_all_results
from streamlit_option_menu import option_menu
from functools import partial
import matplotlib.pyplot as plt
from graphviz import Digraph
import logging
import sys
import validators
from typing import Any, List, Dict


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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

# Configure Streamlit Page
st.set_page_config(
    layout="wide",
    page_title="RAGExplorer",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

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
        "api_key": "",  # Initialized as empty string
        "recommended_splitter": None,
        "use_hyde": False,
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Caching Expensive Functions
@st.cache_data(show_spinner=False)
def cached_load_pdf_data(pdf_paths: List[str], method: str) -> Any:
    return load_pdf_data(pdf_paths, method=method)

@st.cache_resource(show_spinner=False)
def cached_create_vectorstore(
    _splits: List[Any], batch_size: int, use_hyde: bool, api_key: str
) -> Any:
    return create_vectorstore(
        _splits, batch_size=batch_size, use_hyde=use_hyde, api_key=api_key
    )

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
                    st.image(img, caption=f"Image {j+1}", use_column_width=True)

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

    st.subheader("Chunk Size Distribution")
    hist_data = []
    group_labels = []
    for splitter, result in results.items():
        chunk_sizes = [len(chunk.page_content) for chunk in result['splits']]
        hist_data.append(chunk_sizes)
        group_labels.append(splitter)

    fig = ff.create_distplot(hist_data, group_labels, bin_size=100)
    fig.update_layout(
        title_text='Distribution of Chunk Sizes',
        xaxis_title_text='Chunk Size (characters)',
        yaxis_title_text='Density'
    )
    st.plotly_chart(fig)

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

# Page: Home
def home_page() -> None:
    st.header("Welcome to RAGExplorer")
    st.write(
        "This application helps you understand Retrieval-Augmented Generation (RAG) and its internal workings."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What is RAG?")
        st.write(
            """
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with 
            external knowledge retrieval. It allows AI models to access and utilize information beyond their training data, 
            leading to more accurate and up-to-date responses.
            """
        )
    with col2:
        st.subheader("How to use this app")
        st.write(
            """
            1. **Load PDFs**: Start by uploading your PDF documents.
            2. **Text Splitting**: Experiment with different text splitting techniques.
            3. **Retriever Methods**: Explore various retriever options.
            4. **RAG Chain**: Test the RAG chain with your own queries.
            5. **Analyze Results**: Examine the generated results and visualizations.
            """
        )

    if st.button("Start Exploring", key="start_exploring_home"):
        st.session_state.current_step = 1
        st.rerun()

# Page: PDF Loading
def pdf_loading_page() -> None:
    st.header("📄 PDF Loading")

    pdf_source = st.radio(
        "Choose PDF source:",
        ["Upload PDFs", "Use provided PDF", "Use built-in link", "Use custom URLs"],
        horizontal=True,
    )

    pdf_paths = []

    if pdf_source == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True
        )
        if uploaded_files:
            pdf_paths = process_uploaded_files(uploaded_files)
            st.success(f"{len(uploaded_files)} file(s) uploaded")
    elif pdf_source == "Use provided PDF":
        pdf_paths = ["LargeLanguageModel.pdf"]
        st.info(f"Using provided PDF: {pdf_paths[0]}")
    elif pdf_source == "Use built-in link":
        pdf_paths = ["https://arxiv.org/pdf/1706.03762.pdf"]
        st.info(f"Using built-in link: {pdf_paths[0]}")
    else:  # Use custom URLs
        raw_urls = st.text_area("Enter PDF URLs (one per line):").split("\n")
        pdf_paths = [url.strip() for url in raw_urls if url.strip()]
        if pdf_paths:
            valid_pdf_paths = validate_pdf_urls(pdf_paths)
            if len(valid_pdf_paths) < len(pdf_paths):
                st.warning("Some URLs may not be valid PDFs and have been excluded.")
            pdf_paths = valid_pdf_paths
        else:
            st.info("Please enter valid PDF URLs ending with '.pdf'")

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
                    data = cached_load_pdf_data(
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

# Page: Text Splitting
def text_splitting_page() -> None:
    st.header("✂️ Text Splitting")

    if not st.session_state.get("pdf_loaded", False):
        st.warning("Please load a PDF first in the PDF Loading step.")
        return

    st.info(
        """
        Text splitting is crucial for processing large documents. It breaks the text into smaller, manageable chunks.
        Different splitters work better for different types of documents. Experiment to find the best one for your needs.
        """
    )

    splitter_types = {
        "Character": "Splits text based on a fixed number of characters. Simple but may break words.",
        "Recursive": "Intelligently splits text into chunks, trying to keep sentences and paragraphs intact.",
        "Token": "Splits text based on the number of tokens (words or subwords). Useful for maintaining context.",
        "Semantic": "Uses embeddings to split text based on semantic meaning. Requires OpenAI API key.",
    }

    selected_splitters = st.multiselect(
        "Choose splitter type(s):",
        list(splitter_types.keys()),
        default=["Character"],
        help="Select one or more splitters to compare different splitting techniques.",
    )

    for splitter, description in splitter_types.items():
        if splitter in selected_splitters:
            st.markdown(f"**{splitter} Splitter**: {description}")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "Chunk size",
            100,
            2000,
            1000,
            help="Number of characters/tokens per chunk. Larger sizes retain more context but may be less manageable.",
        )
    with col2:
        chunk_overlap = st.slider(
            "Chunk overlap",
            0,
            500,
            200,
            help="Number of overlapping characters/tokens between chunks to preserve context across splits.",
        )

    semantic_params = {}
    if "Semantic" in selected_splitters:
        if not st.session_state.api_key:
            st.error("❗ OpenAI API Key is required for the Semantic Splitter. Please provide it in the sidebar.")
            return
        with st.expander("Semantic Splitter Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                semantic_params["threshold_type"] = st.selectbox(
                    "Breakpoint Threshold Type",
                    ["percentile", "standard_deviation", "interquartile"],
                    help="Method to determine breakpoints in the text",
                )
            with col2:
                semantic_params["threshold_amount"] = st.slider(
                    "Breakpoint Threshold Amount",
                    0.0,
                    1.0,
                    0.95,
                    0.01,
                    help="Threshold for determining breakpoints",
                )
            semantic_params["number_of_chunks"] = st.slider(
                "Number of Chunks",
                10,
                1000,
                500,
                help="Target number of chunks to create",
            )

    if st.button("Process PDF", key="process_pdf_button"):
        if "Semantic" in selected_splitters and not st.session_state.api_key:
            st.error("❗ OpenAI API Key is required for the Semantic Splitter.")
            return
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
                st.session_state.recommended_splitter = min(
                    results, key=lambda x: results[x]["coefficient_of_variation"]
                )
            except Exception as e:
                handle_error(e, "Error processing PDF")

# Page: Retriever
def retriever_page() -> None:
    st.header("🔍 Vector Store and Retriever")

    if "split_results" not in st.session_state:
        st.warning("Please process the PDF in the Text Splitting step first.")
        return

    if "recommended_splitter" not in st.session_state:
        st.warning("No recommended splitter found. Please complete the Text Splitting step.")
        return

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
                vectorstore = cached_create_vectorstore(
                    _splits=splits,  # Changed from splits to _splits
                    batch_size=batch_size,
                    use_hyde=use_hyde,
                    api_key=st.session_state.get("api_key"),
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.use_hyde = use_hyde
                st.success(
                    f"{'HYDE ' if use_hyde else ''}Vector Store created successfully!"
                )
            except Exception as e:
                handle_error(e, "Error creating vector store")

    if "vectorstore" in st.session_state:
        st.subheader("Retriever Settings")
        retriever_type = st.selectbox(
            "Retriever type",
            ["Vector Store", "Reranker", "Hybrid", "Hybrid Reranker"],
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

        if retriever_type in ["Reranker", "Hybrid Reranker"]:
            reranker_type = st.selectbox(
                "Reranker type", ["huggingface", "cohere"], key="reranker_type"
            )
            top_n = st.slider(
                "Number of documents to rerank (top_n)", 1, 10, 3, key="reranker_top_n"
            )

        if retriever_type in ["Hybrid", "Hybrid Reranker"]:
            bm25_weight = st.slider(
                "BM25 Weight",
                0.0,
                1.0,
                0.5,
                0.1,
                key="bm25_weight",
                help="Weight for BM25 retrieval in the hybrid retriever.",
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

                    if retriever_type in ["Reranker", "Hybrid Reranker"]:
                        reranker_retriever = get_reranker_retriever(
                            base_retriever, reranker_type, top_n
                        )
                        st.session_state.reranker_retriever = reranker_retriever

                    if retriever_type in ["Hybrid", "Hybrid Reranker"]:
                        hybrid_retriever = get_hybrid_retriever(
                            st.session_state.split_results[
                                st.session_state.recommended_splitter
                            ],
                            base_retriever,
                            bm25_weight,
                            vector_weight,
                        )
                        st.session_state.hybrid_retriever = hybrid_retriever

                    if retriever_type == "Hybrid Reranker":
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
                        st.session_state.hybrid_reranker_retriever = (
                            hybrid_reranker_retriever
                        )

                    st.session_state.current_retriever_type = retriever_type
                    st.success(
                        f"{'HYDE ' if st.session_state.use_hyde else ''}Vector Store and {retriever_type} Retriever created successfully!"
                    )
                except Exception as e:
                    handle_error(e, "Error creating retriever")

        if "base_retriever" in st.session_state:
            st.subheader("Test Retriever")
            query = st.text_input(
                "Enter a query to test the retriever:", key="retriever_query"
            )
            if st.button("Retrieve", key="retrieve_button"):
                if not query.strip():
                    st.error("❗ Please enter a valid query.")
                else:
                    st.session_state.query = query
                    st.session_state.retrieval_done = True
                    st.rerun()

            if st.session_state.get("retrieval_done", False):
                query = st.session_state.query
                with st.spinner("Retrieving documents..."):
                    try:
                        base_docs = retrieve_documents(
                            st.session_state.base_retriever, query
                        )

                        comparison_docs = []
                        if st.session_state.current_retriever_type != "Vector Store":
                            if st.session_state.current_retriever_type == "Reranker":
                                comparison_docs = retrieve_documents(
                                    st.session_state.reranker_retriever, query
                                )
                            elif st.session_state.current_retriever_type == "Hybrid":
                                comparison_docs = retrieve_documents(
                                    st.session_state.hybrid_retriever, query
                                )
                            elif st.session_state.current_retriever_type == "Hybrid Reranker":
                                comparison_docs = retrieve_documents(
                                    st.session_state.hybrid_reranker_retriever, query
                                )

                        st.subheader("Retrieved Documents")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### Base Retriever")
                            display_retrieved_docs(base_docs, "Base Retriever")

                        with col2:
                            if st.session_state.current_retriever_type != "Vector Store":
                                st.markdown(
                                    f"### {st.session_state.current_retriever_type}"
                                )
                                display_retrieved_docs(
                                    comparison_docs, st.session_state.current_retriever_type
                                )
                            else:
                                st.markdown("### Comparison")
                                st.info(
                                    "No comparison retriever selected. Using Vector Store only."
                                )

                        st.subheader("Embedding Visualization")
                        method = st.selectbox(
                            "Choose visualization method", ["pacmap", "umap", "tsne"]
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
                                    retrieved_docs = {"Base": base_docs}
                                    if st.session_state.current_retriever_type != "Vector Store":
                                        retrieved_docs[
                                            st.session_state.current_retriever_type
                                        ] = comparison_docs

                                    fig = create_embedding_visualization(
                                        st.session_state.vectorstore,
                                        st.session_state.split_results[
                                            st.session_state.recommended_splitter
                                        ],
                                        query,
                                        retrieved_docs,
                                        method=method,
                                        sample_size=sample_size if sample_size > 0 else None,
                                        dimensions=dimensions,
                                        api_key=st.session_state.get("api_key"),
                                    )
                                    st.plotly_chart(fig)
                                except Exception as e:
                                    handle_error(e, "Error creating embedding visualization")
                    except Exception as e:
                        handle_error(e, "Error retrieving documents")

# Page: RAG Chain
def rag_chain_page() -> None:
    st.header("🔗 RAG Chain")

    if "vectorstore" not in st.session_state:
        st.warning("Please create a Vector Store in the Retriever step first.")
        return

    if not st.session_state.api_key:
        st.warning("Please provide your OpenAI API Key in the sidebar to use the RAG Chain.")
        return

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

# Check if user can proceed to the next step
def can_proceed(current_step: int) -> bool:
    if current_step == 0:  # Home page
        return True
    elif current_step == 1:  # PDF Loading
        return st.session_state.get("pdf_loaded", False)
    elif current_step == 2:  # Text Splitting
        return "split_results" in st.session_state
    elif current_step == 3:  # Retriever
        return "vectorstore" in st.session_state and st.session_state.get("api_key")
    else:
        return st.session_state.get("api_key")

# Main Function
def main() -> None:
    logger.info("Application started.")

    st.title("🔍 RAGExplorer")
    st.markdown("Explore the inner workings of Retrieval-Augmented Generation (RAG)")

    # Add OpenAI API Key input in the sidebar
    with st.sidebar:
        st.markdown("## OpenAI API Key")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if api_key:
            st.success("API Key provided!")
            st.session_state.api_key = api_key  # Store the API key in session state
        else:
            st.warning("Please enter your OpenAI API Key to use the application.")

        st.markdown("---")

        # Sidebar Navigation
        st.markdown("## Navigation")
        selected = option_menu(
            menu_title=None,
            options=["Home", "PDF Loading", "Text Splitting", "Retriever", "RAG Chain"],
            icons=["house", "file-pdf", "scissors", "search", "link"],
            menu_icon="cast",
            default_index=st.session_state.current_step,
        )

        # Add some space
        st.markdown("---")

        # Steps and Progress Indicator
        steps = ["Home", "PDF Loading", "Text Splitting", "Retriever", "RAG Chain"]
        current_step = steps.index(selected)
        st.progress((current_step + 1) / len(steps), f"Step {current_step + 1} of {len(steps)}")

        # About Section
        with st.expander("About RAGExplorer"):
            st.write(
                "RAGExplorer helps you understand the inner workings of Retrieval-Augmented Generation."
            )
            st.write(
                "Navigate through the steps to explore each component of the RAG process."
            )

    # Display the selected page
    pages = {
        "Home": home_page,
        "PDF Loading": pdf_loading_page,
        "Text Splitting": text_splitting_page,
        "Retriever": retriever_page,
        "RAG Chain": rag_chain_page,
    }
    pages[selected]()

    # Navigation Buttons (Optional - Since navigation is handled via sidebar)
    # This section can be removed or retained based on user preference.
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_step > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_step -= 1
                st.rerun()

    with col3:
        if current_step < len(steps) - 1:
            if st.button("Next ➡️"):
                if can_proceed(current_step):
                    st.session_state.current_step += 1
                    st.rerun()
                else:
                    st.error("Please complete the current step before proceeding.")

if __name__ == "__main__":
    main()