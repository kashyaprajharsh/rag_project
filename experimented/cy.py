import streamlit as st
import extra_streamlit_components as stx
from streamlit_option_menu import option_menu
from pdf import load_pdf_data, process_uploaded_file, split_text
from PIL import Image
import io
import plotly.graph_objects as go
import difflib
import numpy as np
from itertools import combinations
from vectordb import create_vectorstore, get_retriever, get_reranker_retriever, get_hybrid_retriever, get_hybrid_reranker_retriever, retrieve_documents
from rag import generate_all_rag_answers, get_rag_chain, RAG_TYPES

# Set page config
st.set_page_config(layout="wide", page_title="RAGExplorer", page_icon="🔍")

# Custom CSS for improved styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #50E3C2;
        --background-color: #F5F7FA;
        --text-color: #333333;
        --accent-color: #FF6B6B;
    }
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3A7BC8;
    }
    h1, h2, h3 {
        color: var(--primary-color);
    }
    .stAlert {
        background-color: var(--secondary-color);
        color: var(--text-color);
    }
    /* Card styling */
    .css-1y4p8pa {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFFFFF;
    }
    /* Custom classes */
    .custom-tab {
        background-color: var(--primary-color);
        color: white;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
        margin-right: 2px;
    }
    .custom-tab-content {
        background-color: white;
        padding: 20px;
        border-radius: 0 5px 5px 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

def main():
    # Sidebar
    with st.sidebar:
        #st.image("logo.png", width=100)  # Add your logo image
        st.title("RAGExplorer")
        
        # Navigation using option_menu
        selected = option_menu(
            menu_title=None,
            options=["PDF Loading", "Text Splitting", "Retriever", "RAG Chain"],
            icons=["file-earmark-pdf", "scissors", "search", "link"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.session_state.current_step = ["PDF Loading", "Text Splitting", "Retriever", "RAG Chain"].index(selected)

    # Main content
    if st.session_state.current_step == 0:
        pdf_loading()
    elif st.session_state.current_step == 1:
        text_splitting()
    elif st.session_state.current_step == 2:
        retriever()
    elif st.session_state.current_step == 3:
        rag_chain()

import extra_streamlit_components as stx

def pdf_loading():
    st.header("📥 PDF Loading")
    
    # Create TabBarItemData objects for the tab bar
    tabs = [
        stx.TabBarItemData(id="upload", title="Upload PDF", description="Upload a PDF file"),
        stx.TabBarItemData(id="provided", title="Use provided PDF", description="Use a pre-loaded PDF"),
        stx.TabBarItemData(id="builtin", title="Use built-in link", description="Use a built-in PDF link"),
        stx.TabBarItemData(id="custom", title="Paste custom link", description="Use a custom PDF link")
    ]

    # Create the tab bar
    chosen_tab = stx.tab_bar(tabs)

    if chosen_tab == "upload":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            pdf_path = process_uploaded_file(uploaded_file)
        else:
            pdf_path = None
    elif chosen_tab == "provided":
        pdf_path = "ASIANPAINT_jul23.pdf"
    elif chosen_tab == "builtin":
        pdf_path = "https://arxiv.org/pdf/1706.03762"
    else:  # custom link
        pdf_path = st.text_input("Enter the PDF link:")

    loading_method = st.selectbox("Loading method:", ["Normal", "With Images"])
    
    if st.button("Load PDF", key="load_pdf"):
        if pdf_path:
            with st.spinner("Loading PDF..."):
                try:
                    if pdf_path.startswith('http'):
                        data = load_pdf_data(pdf_path, method='normal')
                    else:
                        data = load_pdf_data(pdf_path, method=loading_method.lower().replace(' ', '_'))
                    st.session_state['pdf_data'] = data
                    st.session_state['pdf_loaded'] = True
                    st.success("PDF loaded successfully!")
                    
                    # Display PDF content
                    with st.expander("View PDF Content"):
                        for i, page in enumerate(data):
                            st.subheader(f"Page {i+1}")
                            st.text_area(f"Content", page.page_content, height=200)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.json(page.metadata)
                            
                            with col2:
                                if loading_method == "With Images" and 'images' in page.metadata:
                                    for j, img_data in enumerate(page.metadata['images']):
                                        img = Image.open(io.BytesIO(img_data))
                                        st.image(img, caption=f"Image {j+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading PDF: {str(e)}")
        else:
            st.error("Please provide a valid PDF source.")


def text_splitting():
    st.header("✂️ Text Splitting")
    
    if 'pdf_loaded' not in st.session_state or not st.session_state['pdf_loaded']:
        st.warning("Please load a PDF first.")
        return

    st.info("""
    Text splitting is crucial for processing large documents. It breaks the text into smaller, manageable chunks.
    Different splitters work better for different types of documents. Experiment to find the best one for your needs.
    """)

    splitter_types = {
        "Character": "Splits text based on a fixed number of characters. Simple but may break words.",
        "Recursive": "Intelligently splits text into chunks, trying to keep sentences and paragraphs intact.",
        "Token": "Splits text based on the number of tokens (words or subwords). Useful for maintaining context.",
        "Semantic": "Uses embeddings to split text based on semantic meaning. Requires OpenAI API key."
    }

    selected_splitters = st.multiselect(
        "Choose splitter type(s):", 
        list(splitter_types.keys()), 
        default=["Character"],
        help="Select one or more splitters to compare"
    )

    for splitter, description in splitter_types.items():
        if splitter in selected_splitters:
            st.markdown(f"**{splitter} Splitter**: {description}")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size", 100, 2000, 1000, help="Number of characters/tokens per chunk")
    with col2:
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, help="Number of overlapping characters/tokens between chunks")

    semantic_params = {}
    if "Semantic" in selected_splitters:
        st.subheader("Semantic Splitter Parameters")
        col1, col2 = st.columns(2)
        with col1:
            semantic_params['threshold_type'] = st.selectbox(
                "Breakpoint Threshold Type",
                ["percentile", "standard_deviation", "interquartile"],
                help="Method to determine breakpoints in the text"
            )
        with col2:
            semantic_params['threshold_amount'] = st.slider(
                "Breakpoint Threshold Amount",
                0.0, 1.0, 0.95, 0.01,
                help="Threshold for determining breakpoints"
            )
        semantic_params['number_of_chunks'] = st.slider(
            "Number of Chunks",
            10, 1000, 500,
            help="Target number of chunks to create"
        )

    if st.button("Process PDF", key="process_pdf"):
        with st.spinner("Processing..."):
            try:
                split_results = {}
                processing_times = {}
                for splitter in selected_splitters:
                    if splitter.lower() == 'semantic':
                        split_data, processing_time = split_text(st.session_state['pdf_data'], splitter.lower(), semantic_params=semantic_params)
                    else:
                        split_data, processing_time = split_text(st.session_state['pdf_data'], splitter.lower(), chunk_size, chunk_overlap)
                    split_results[splitter] = split_data
                    processing_times[splitter] = processing_time
                    st.success(f"{splitter} splitter: Text split into {len(split_data)} chunks in {processing_time:.2f} seconds")

                # Visualize chunk sizes
                st.subheader("Chunk Size Distribution")
                data = [go.Box(y=[len(chunk.page_content) for chunk in chunks], name=splitter, boxpoints='all', jitter=0.3, pointpos=-1.8) for splitter, chunks in split_results.items()]
                fig = go.Figure(data=data)
                fig.update_layout(
                    title='Distribution of Chunk Sizes',
                    yaxis_title='Chunk Size (characters)',
                    boxmode='group',
                    height=600,
                    showlegend=True,
                    template="plotly_white"
                )
                st.plotly_chart(fig)

                # Splitter recommendation
                st.subheader("Splitter Recommendation")
                chunk_std_devs = {splitter: np.std([len(chunk.page_content) for chunk in chunks]) for splitter, chunks in split_results.items()}
                recommended_splitter = min(chunk_std_devs, key=chunk_std_devs.get)
                st.success(f"Based on chunk size consistency, the recommended splitter is: **{recommended_splitter}**")
                st.info(f"The {recommended_splitter} splitter produced the most consistent chunk sizes, which is often desirable for even processing.")
                st.info(f"However, consider the trade-off between consistency and processing time. {recommended_splitter} took {processing_times[recommended_splitter]:.2f} seconds to process.")

                # Compare splitters
                if len(selected_splitters) > 1:
                    st.subheader("Splitter Comparison")
                    
                    # Determine the number of chunks to display
                    num_chunks = min(5, min(len(split_results[splitter]) for splitter in selected_splitters))
                    
                    for i in range(num_chunks):
                        with st.expander(f"Chunk {i+1}"):
                            # Create tabs for different views
                            tab_full, tab_diff = st.tabs(["Full Content", "Differences"])
                            
                            with tab_full:
                                # Display full content of each chunk
                                for splitter in selected_splitters:
                                    st.markdown(f"**{splitter} splitter:**")
                                    st.text_area(
                                        f"{splitter} chunk content",
                                        split_results[splitter][i].page_content,
                                        height=200,
                                        key=f"{splitter}_chunk_{i}"  # Unique key for each text area
                                    )
                            
                            with tab_diff:
                                # Add color legend and explanation
                                st.markdown("""
                                **Color Legend:**
                                <span style="background-color: #aaffaa;">Green</span>: Text present in the second splitter but not in the first
                                <span style="background-color: #ffaaaa;">Red</span>: Text present in the first splitter but not in the second
                                <span style="background-color: #ffffaa;">Yellow</span>: Minor changes or differences in whitespace
                                
                                The differences shown below compare each pair of splitters. Lines without coloring are identical between the two splitters being compared.
                                """, unsafe_allow_html=True)
                                
                                st.markdown("---")

                                # Create pairwise comparisons
                                for j, (splitter1, splitter2) in enumerate(combinations(selected_splitters, 2)):
                                    st.markdown(f"**Difference between {splitter1} and {splitter2}:**")
                                    
                                    d = difflib.Differ()
                                    diff = list(d.compare(split_results[splitter1][i].page_content.splitlines(), 
                                                          split_results[splitter2][i].page_content.splitlines()))
                                    
                                    # Color-code the differences
                                    html_diff = []
                                    for line in diff:
                                        if line.startswith('+'):
                                            html_diff.append(f'<span style="background-color: #aaffaa;">{line}</span>')
                                        elif line.startswith('-'):
                                            html_diff.append(f'<span style="background-color: #ffaaaa;">{line}</span>')
                                        elif line.startswith('?'):
                                            html_diff.append(f'<span style="background-color: #ffffaa;">{line}</span>')
                                        else:
                                            html_diff.append(line)
                                    
                                    st.markdown('<br>'.join(html_diff), unsafe_allow_html=True)
                                    st.markdown("---")

                st.session_state['split_results'] = split_results
                st.session_state['recommended_splitter'] = recommended_splitter

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

def retriever():
    st.header("🔍 Vector Store and Retriever")

    if 'split_results' not in st.session_state:
        st.warning("Please process the PDF in the Text Splitting tab first.")
        return

    st.subheader("Text Splitting")
    selected_splitter = st.selectbox(
        "Choose splitter to use:", 
        list(st.session_state['split_results'].keys()),
        index=list(st.session_state['split_results'].keys()).index(st.session_state['recommended_splitter']),
        help="Select the text splitter to use for creating the vector store"
    )
    
    st.info(f"Using {len(st.session_state['split_results'][selected_splitter])} chunks from the {selected_splitter} splitter.")

    st.subheader("Vector Store")
    use_hyde = st.checkbox("Use HYDE (Hypothetical Document Embedder)", value=False)
    batch_size = st.number_input("Batch size", min_value=100, max_value=10000, value=5000, step=100)
    
    if st.button("Create Vector Store"):
        with st.spinner("Creating Vector Store..."):
            try:
                splits = st.session_state['split_results'][st.session_state['recommended_splitter']]
                vectorstore = create_vectorstore(
                    splits,
                    batch_size=batch_size,
                    use_hyde=use_hyde
                )
                st.session_state['vectorstore'] = vectorstore
                st.session_state['use_hyde'] = use_hyde
                st.success(f"{'HYDE ' if use_hyde else ''}Vector Store created successfully!")
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")
    
    if 'vectorstore' in st.session_state:
        st.subheader("Retriever Settings")
        retriever_type = st.selectbox("Retriever type", ["Vector Store", "Reranker", "Hybrid", "Hybrid Reranker"],
                                      key="retriever_type")
        
        search_type = st.selectbox("Search type", ["similarity", "mmr"],
                                   key="retriever_search_type")
        k = st.slider("Number of documents to retrieve (k)", 1, 20, 5,
                      key="retriever_k")
        if search_type == "mmr":
            fetch_k = st.slider("Number of documents to fetch for MMR (fetch_k)", k, 50, 20,
                                key="retriever_fetch_k")
        else:
            fetch_k = k
        
        if retriever_type in ["Reranker", "Hybrid Reranker"]:
            reranker_type = st.selectbox("Reranker type", ["huggingface", "cohere"],
                                         key="reranker_type")
            top_n = st.slider("Number of documents to rerank (top_n)", 1, 10, 3,
                              key="reranker_top_n")
        
        if retriever_type in ["Hybrid", "Hybrid Reranker"]:
            bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.5, 0.1,
                                    key="bm25_weight")
            vector_weight = 1 - bm25_weight
            st.info(f"Vector Store Weight: {vector_weight:.1f}")
        
        if st.button("Create Retriever"):
            with st.spinner("Creating Retriever..."):
                try:
                    base_retriever = get_retriever(st.session_state['vectorstore'], search_type, k, fetch_k)
                    st.session_state['base_retriever'] = base_retriever
                    
                    if retriever_type in ["Reranker", "Hybrid Reranker"]:
                        reranker_retriever = get_reranker_retriever(base_retriever, reranker_type, top_n)
                        st.session_state['reranker_retriever'] = reranker_retriever
                    
                    if retriever_type in ["Hybrid", "Hybrid Reranker"]:
                        hybrid_retriever = get_hybrid_retriever(st.session_state['split_results'][st.session_state['recommended_splitter']], 
                                                                base_retriever, bm25_weight, vector_weight)
                        st.session_state['hybrid_retriever'] = hybrid_retriever
                    
                    if retriever_type == "Hybrid Reranker":
                        hybrid_reranker_retriever = get_hybrid_reranker_retriever(
                            st.session_state['split_results'][st.session_state['recommended_splitter']],
                            base_retriever, bm25_weight, vector_weight, reranker_type, top_n
                        )
                        st.session_state['hybrid_reranker_retriever'] = hybrid_reranker_retriever
                    
                    st.session_state['current_retriever_type'] = retriever_type
                    st.success(f"{'HYDE ' if st.session_state['use_hyde'] else ''}Vector Store and {retriever_type} Retriever created successfully!")
                except Exception as e:
                    st.error(f"Error creating retriever: {str(e)}")
    
        if 'base_retriever' in st.session_state:
            st.subheader("Test Retriever")
            query = st.text_input("Enter a query to test the retriever:")
            if st.button("Retrieve"):
                with st.spinner("Retrieving documents..."):
                    try:
                        base_docs = retrieve_documents(st.session_state['base_retriever'], query)
                        
                        current_retriever_type = st.session_state.get('current_retriever_type', 'Vector Store')
                        
                        if current_retriever_type in ["Reranker", "Hybrid", "Hybrid Reranker"]:
                            if current_retriever_type == "Reranker":
                                comparison_docs = retrieve_documents(st.session_state['reranker_retriever'], query)
                                comparison_type = f"Reranked ({st.session_state.get('reranker_type', 'unknown')})"
                            elif current_retriever_type == "Hybrid":
                                comparison_docs = retrieve_documents(st.session_state['hybrid_retriever'], query)
                                comparison_type = "Hybrid"
                            else:  # Hybrid Reranker
                                comparison_docs = retrieve_documents(st.session_state['hybrid_reranker_retriever'], query)
                                comparison_type = f"Hybrid Reranked ({st.session_state.get('reranker_type', 'unknown')})"
                            
                            st.subheader("Comparison of Results")
                            
                            tab1, tab2, tab3 = st.tabs(["Side by Side", "Differences", "Hybrid Comparison"])
                            
                            with tab1:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Base Vector Store Results**")
                                    for i, doc in enumerate(base_docs):
                                        with st.expander(f"Document {i+1}"):
                                            st.markdown(f"**Content:**\n{doc.page_content}")
                                            st.markdown("**Metadata:**")
                                            st.json(doc.metadata)
                            
                                with col2:
                                    st.markdown(f"**{comparison_type} Results**")
                                    for i, doc in enumerate(comparison_docs):
                                        with st.expander(f"Document {i+1}"):
                                            st.markdown(f"**Content:**\n{doc.page_content}")
                                            st.markdown("**Metadata:**")
                                            st.json(doc.metadata)
                            
                            with tab2:
                                st.markdown(f"**Differences between Base Vector Store and {comparison_type} Results**")
                                for i in range(min(len(base_docs), len(comparison_docs))):
                                    with st.expander(f"Document {i+1}"):
                                        d = difflib.Differ()
                                        diff = list(d.compare(base_docs[i].page_content.splitlines(), 
                                                            comparison_docs[i].page_content.splitlines()))
                                        
                                        html_diff = []
                                        for line in diff:
                                            if line.startswith('+'):
                                                html_diff.append(f'<span style="background-color: #aaffaa;">{line}</span>')
                                            elif line.startswith('-'):
                                                html_diff.append(f'<span style="background-color: #ffaaaa;">{line}</span>')
                                            elif line.startswith('?'):
                                                html_diff.append(f'<span style="background-color: #ffffaa;">{line}</span>')
                                            else:
                                                html_diff.append(line)
                                        
                                        st.markdown('<br>'.join(html_diff), unsafe_allow_html=True)
                            
                            with tab3:
                                if current_retriever_type == "Hybrid Reranker" and 'hybrid_retriever' in st.session_state:
                                    hybrid_docs = retrieve_documents(st.session_state['hybrid_retriever'], query)
                                    st.markdown("**Hybrid vs Hybrid Reranker Comparison**")
                                    for i in range(min(len(hybrid_docs), len(comparison_docs))):
                                        with st.expander(f"Document {i+1}"):
                                            st.markdown("**Hybrid Result:**")
                                            st.markdown(hybrid_docs[i].page_content)
                                            st.markdown("**Hybrid Reranker Result:**")
                                            st.markdown(comparison_docs[i].page_content)
                                            st.markdown("**Differences:**")
                                            d = difflib.Differ()
                                            diff = list(d.compare(hybrid_docs[i].page_content.splitlines(), 
                                                                comparison_docs[i].page_content.splitlines()))
                                            
                                            html_diff = []
                                            for line in diff:
                                                if line.startswith('+'):
                                                    html_diff.append(f'<span style="background-color: #aaffaa;">{line}</span>')
                                                elif line.startswith('-'):
                                                    html_diff.append(f'<span style="background-color: #ffaaaa;">{line}</span>')
                                                elif line.startswith('?'):
                                                    html_diff.append(f'<span style="background-color: #ffffaa;">{line}</span>')
                                                else:
                                                    html_diff.append(line)
                                            
                                            st.markdown('<br>'.join(html_diff), unsafe_allow_html=True)
                                else:
                                    st.info("Hybrid comparison is only available for Hybrid Reranker retriever type when Hybrid retriever has been created.")
                        
                        else:  # Vector Store only
                            st.subheader("Vector Store Results")
                            for i, doc in enumerate(base_docs):
                                with st.expander(f"Document {i+1}"):
                                    st.markdown(f"**Content:**\n{doc.page_content}")
                                    st.markdown("**Metadata:**")
                                    st.json(doc.metadata)
                    
                    except Exception as e:
                        st.error(f"Error retrieving documents: {str(e)}")

def rag_chain():
    st.header("🔗 RAG Chain")

    if 'vectorstore' not in st.session_state:
        st.warning("Please create a Vector Store in the Retriever tab first.")
        return

    # Initialize chat history and results
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_results" not in st.session_state:
        st.session_state.rag_results = []

    # Display chat messages and results from history
    for i in range(0, len(st.session_state.messages), 2):
        # Display user question
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[i]["content"])
        
        # Display assistant response and results
        if i + 1 < len(st.session_state.messages):
            with st.chat_message("assistant"):
                st.markdown(st.session_state.messages[i + 1]["content"])
            
            if i // 2 < len(st.session_state.rag_results):
                results = st.session_state.rag_results[i // 2]
                st.subheader(f"Results for Question {i//2 + 1}")
                rag_tabs = st.tabs(list(results.keys()))
                for rag_tab, (rag_method, result) in zip(rag_tabs, results.items()):
                    with rag_tab:
                        st.markdown("### Answer")
                        st.write(result['answer'])
                        
                        st.markdown("### Source Context")
                        for j, doc in enumerate(result['context'], 1):
                            with st.expander(f"Source {j}"):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown("**Content:**")
                                    st.markdown(doc.page_content)
                                with col2:
                                    st.markdown("**Metadata:**")
                                    st.json(doc.metadata)

    # Chat input
    prompt = st.chat_input("What is your question?")
    if prompt:
        # Add the new message to the state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate answer immediately
        with st.spinner("Generating answers..."):
            try:
                retrievers = {
                    'base_retriever': st.session_state.get('base_retriever'),
                    'reranker_retriever': st.session_state.get('reranker_retriever'),
                    'hybrid_retriever': st.session_state.get('hybrid_retriever'),
                    'hybrid_reranker_retriever': st.session_state.get('hybrid_reranker_retriever')
                }
                
                results = generate_all_rag_answers(prompt, retrievers)
                
                if results:
                    st.session_state.rag_results.append(results)
                    st.session_state.messages.append({"role": "assistant", "content": "I've generated answers using different RAG methods. Please check the results above."})
                else:
                    st.error("No results were generated. Please check if any retrievers are available and properly configured.")

            except Exception as e:
                st.error(f"Error generating answers: {str(e)}")

        # Force a rerun to display the new message and results immediately
        st.rerun()

if __name__ == "__main__":
    main()