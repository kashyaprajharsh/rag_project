__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain.chains import HypotheticalDocumentEmbedder
import os
import openai
import pacmap
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Callable
from langchain.schema import Document
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from uuid import uuid4

os.environ["COHERE_API_KEY"] = st.secrets['COHERE_API_KEY']

def create_vectorstore(splits, batch_size=5000, use_hyde=False, api_key=None):
    """
    Create a vector store from the given splits.
    
    :param splits: List of Document objects
    :param batch_size: Number of documents to process in each batch
    :param use_hyde: Whether to use HYDE embeddings
    :param api_key: OpenAI API key
    :return: Chroma vector store
    """
    if not api_key:
        raise ValueError("OpenAI API Key is required. Please provide it in the sidebar.")
    
    if use_hyde:
        llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", openai_api_key=api_key)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(llm, embeddings, prompt_key="dbpedia_entity")
        vectorstore = Chroma(embedding_function=hyde_embeddings)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = Chroma(embedding_function=embeddings)
    
    # Generate UUIDs for all documents
    uuids = [str(uuid4()) for _ in range(len(splits))]
    
    # Add unique_id to each document's metadata
    for split, uuid in zip(splits, uuids):
        if not hasattr(split, 'metadata'):
            split.metadata = {}
        split.metadata['unique_id'] = uuid

    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        batch_uuids = uuids[i:i + batch_size]
        vectorstore.add_documents(documents=batch, ids=batch_uuids)
    
    return vectorstore


def get_retriever(vectorstore, search_type="similarity", k=5, fetch_k=20):
    """
    Get a retriever from the given vector store.
    
    :param vectorstore: Chroma vector store
    :param search_type: Type of search to perform ("similarity" or "mmr")
    :param k: Number of documents to retrieve
    :param fetch_k: Number of documents to fetch for MMR
    :return: Retriever object
    """
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
    
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

def get_reranker_retriever(base_retriever, reranker_type="huggingface", top_n=3):
    if reranker_type == "huggingface":
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
    elif reranker_type == "cohere":
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=top_n,cohere_api_key=st.secrets['COHERE_API_KEY'])
    else:
        raise ValueError("Invalid reranker type. Choose 'huggingface' or 'cohere'.")
    
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

def get_hybrid_retriever(splits, vector_retriever, bm25_weight=0.5, vector_weight=0.5):
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = vector_retriever.search_kwargs["k"]
    
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )

def get_hybrid_reranker_retriever(splits, vector_retriever, bm25_weight=0.5, vector_weight=0.5, reranker_type="huggingface", top_n=3):
    hybrid_retriever = get_hybrid_retriever(splits, vector_retriever, bm25_weight, vector_weight)
    
    if reranker_type == "huggingface":
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
    elif reranker_type == "cohere":
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=top_n)
    else:
        raise ValueError("Invalid reranker type. Choose 'huggingface' or 'cohere'.")
    
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid_retriever)

def retrieve_documents(retriever, query):
    """
    Retrieve documents using the given retriever and query.
    
    :param retriever: Retriever object
    :param query: Query string
    :return: List of retrieved documents
    """
    return retriever.invoke(query)


def create_embedding_visualization(vectorstore, docs_processed: List[Document], query: str, retrieved_docs: Dict[str, List[Document]], method='pacmap', sample_size=None, dimensions=2, use_random_state=True, selected_retrievers=None, api_key=None):
    """
    Create a visualization of document embeddings and retrieved documents.
    
    :param vectorstore: The Chroma vector store containing document embeddings
    :param docs_processed: List of all processed documents
    :param query: The user query
    :param retrieved_docs: Dictionary of retrieved documents for each retriever
    :param method: Dimensionality reduction method ('pacmap', 'umap', or 'tsne')
    :param sample_size: Number of documents to sample (None for all)
    :param dimensions: Number of dimensions for the projection (2 or 3)
    :param use_random_state: Whether to use a fixed random state for reproducibility
    :param selected_retrievers: List of retrievers to visualize
    :param api_key: OpenAI API key
    :return: Plotly figure object
    """
    if not api_key:
        raise ValueError("OpenAI API Key is required. Please provide it in the sidebar.")
    
    # Retrieve embeddings and metadata
    result = vectorstore.get(include=["embeddings", "metadatas", "documents"])
    embeddings = result["embeddings"]
    metadatas = result["metadatas"]
    documents = result["documents"]

    # Sample if necessary
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = [embeddings[i] for i in indices]
        metadatas = [metadatas[i] for i in indices]
        documents = [documents[i] for i in indices]

    # Add query embedding
    query_vector = OpenAIEmbeddings(openai_api_key=api_key).embed_query(query)
    all_embeddings = embeddings + [query_vector]

    # Calculate cosine similarities
    similarities = cosine_similarity(embeddings, [query_vector]).flatten()

    # Perform dimensionality reduction
    if method == 'pacmap':
        reducer = pacmap.PaCMAP(n_components=dimensions, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1 if use_random_state else None)
    elif method == 'umap':
        reducer = UMAP(n_components=dimensions, random_state=1 if use_random_state else None)
    elif method == 'tsne':
        reducer = TSNE(n_components=dimensions, random_state=1 if use_random_state else None)
    else:
        raise ValueError("Invalid method. Choose 'pacmap', 'umap', or 'tsne'.")

    projected = reducer.fit_transform(np.array(all_embeddings))

    # Modify the df_data creation to include all documents
    df_data = [
        {
            "x": point[0],
            "y": point[1],
            "z": point[2] if dimensions == 3 else None,
            "source": metadata.get("source", "Unknown").split("/")[-1],
            "extract": doc[:100] + "...",
            "retriever": [],  # Change this to a list
            "cosine_similarity": sim,
            "rank": {},  # Change this to a dictionary
            "unique_id": metadata["unique_id"]  # Add unique ID
        }
        for point, metadata, doc, sim in zip(projected[:-1], metadatas, documents, similarities)
    ]

    # Add query point
    df_data.append({
        "x": projected[-1][0],
        "y": projected[-1][1],
        "z": projected[-1][2] if dimensions == 3 else None,
        "source": "User query",
        "extract": query,
        "retriever": "Query",
        "cosine_similarity": 1.0,
        "rank": None,
        "unique_id": "query"
    })

    # Mark retrieved documents
    for retriever_name, docs in retrieved_docs.items():
        if selected_retrievers and retriever_name not in selected_retrievers:
            continue
        for rank, doc in enumerate(docs, 1):
            matching_docs = [i for i, d in enumerate(df_data) if d['unique_id'] == doc.metadata["unique_id"]]
            for idx in matching_docs:
                if retriever_name not in df_data[idx]["retriever"]:
                    df_data[idx]["retriever"].append(retriever_name)
                df_data[idx]["rank"][retriever_name] = rank

    df = pd.DataFrame(df_data)

    # Create figure
    fig = go.Figure()

    # Add traces for each retriever type
    retriever_types = ['Not retrieved', 'Query'] + (selected_retrievers if selected_retrievers else list(retrieved_docs.keys()))
    for retriever in retriever_types:
        if retriever == "Not retrieved":
            subset = df[~df['retriever'].apply(lambda x: bool(x))]
        elif retriever == "Query":
            subset = df[df['retriever'] == "Query"]
        else:
            subset = df[df['retriever'].apply(lambda x: retriever in x)]
        
        marker_size = 30 if retriever == "Query" else (10 if retriever == "Not retrieved" else 15)
        symbol = "diamond-open" if retriever == "Query" else "circle"
        
        if dimensions == 2:
            fig.add_trace(go.Scatter(
                x=subset['x'], y=subset['y'],
                mode='markers',
                marker=dict(size=marker_size, symbol=symbol),
                name=retriever,
                text=[f"Rank: {rank}<br>Cosine Similarity: {sim:.3f}<br>Source: {source}<br>Extract: {extract}" 
                      for rank, sim, source, extract in zip(subset['rank'], subset['cosine_similarity'], subset['source'], subset['extract'])],
                hoverinfo="text+name"
            ))
        else:  # 3D
            fig.add_trace(go.Scatter3d(
                x=subset['x'], y=subset['y'], z=subset['z'],
                mode='markers',
                marker=dict(size=marker_size, symbol=symbol),
                name=retriever,
                text=[f"Rank: {rank}<br>Cosine Similarity: {sim:.3f}<br>Source: {source}<br>Extract: {extract}" 
                      for rank, sim, source, extract in zip(subset['rank'], subset['cosine_similarity'], subset['source'], subset['extract'])],
                hoverinfo="text+name"
            ))

    # Update layout
    fig.update_layout(
        title=f"<b>{dimensions}D Projection of Document Embeddings via {method.upper()}</b>",
        legend_title="<b>Retriever</b>",
        hovermode="closest",
        width=1000,
        height=700
    )

    if dimensions == 2:
        fig.update_xaxes(title="Dimension 1")
        fig.update_yaxes(title="Dimension 2")
    else:
        fig.update_scenes(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        )

    return fig