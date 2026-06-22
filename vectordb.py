"""Vector store + retrievers, migrated to Qdrant (in-memory) on LangChain v1.

- ChromaDB is replaced by an in-memory ``QdrantVectorStore`` (no server, no disk)
  holding dense embeddings.
- Hybrid retrieval stays **explicit and teachable**: a standalone ``BM25Retriever``
  (lexical) is fused with the dense Qdrant retriever by ``WeightedEnsembleRetriever``,
  a small reciprocal-rank-fusion retriever built on ``langchain_core`` so the
  ``bm25_weight`` / ``vector_weight`` sliders remain meaningful for the demo.
- Reranking uses a custom ``BaseRetriever`` wrapping Cohere / HuggingFace
  cross-encoders. The legacy ``ContextualCompressionRetriever`` /
  ``HypotheticalDocumentEmbedder`` (langchain_classic-only) are intentionally avoided.
- HyDE is a tiny ``Embeddings`` wrapper instead of the legacy embedder.
"""

import os
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pacmap
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from rag import get_llm, extract_content

# Surface the Cohere key (used by CohereRerank) without breaking non-Streamlit
# imports. Only set it when non-empty — an empty value makes the Cohere client
# send an "Authorization: Bearer " header, which fails as an illegal header.
try:
    _cohere_key = st.secrets.get("COHERE_API_KEY", "")
    if _cohere_key:
        os.environ["COHERE_API_KEY"] = _cohere_key
except Exception:
    pass


def _get_cohere_key() -> str:
    try:
        return st.secrets.get("COHERE_API_KEY", "") or os.environ.get("COHERE_API_KEY", "")
    except Exception:
        return os.environ.get("COHERE_API_KEY", "")


def get_embeddings(api_key=None, provider="Google Gemini"):
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API Key is required. Please provide it in the sidebar.")
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    else:  # Google Gemini
        kwargs = {"model": "gemini-embedding-2"}
        if api_key:
            kwargs["google_api_key"] = api_key
        return GoogleGenerativeAIEmbeddings(**kwargs)


class HyDEEmbeddings(Embeddings):
    """Hypothetical Document Embeddings.

    Documents are embedded normally; a *query* is first expanded into a
    hypothetical answer document by ``hyde_chain`` and that text is embedded.
    Replaces the classic ``HypotheticalDocumentEmbedder``.
    """

    def __init__(self, hyde_chain: Any, base_embeddings: Embeddings):
        self.hyde_chain = hyde_chain
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        hypothetical = self.hyde_chain.invoke({"question": text})
        if hasattr(hypothetical, "content"):
            hypothetical = hypothetical.content
        return self.base_embeddings.embed_query(str(hypothetical))


def create_vectorstore(splits, batch_size=5000, use_hyde=False, api_key=None):
    """Create an in-memory Qdrant (dense) vector store from splits.

    :param splits: List of Document objects
    :param batch_size: Number of documents to index per batch
    :param use_hyde: Whether to use HyDE query embeddings
    :param api_key: API key (required only for the OpenAI provider)
    :return: QdrantVectorStore
    """
    provider = st.session_state.get("llm_provider", "Google Gemini")
    base_embeddings = get_embeddings(api_key=api_key, provider=provider)

    if use_hyde:
        llm = get_llm(api_key=api_key, provider=provider)
        hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Generate a concise, focused document that directly answers the given question.
            The document should:
            - Be specific and factual
            - Include key terms and relevant context
            - Be 3-5 sentences long
            - Focus only on information needed to answer the question

            Question: {question}
            Hypothetical Document:""",
        )
        hyde_chain = hyde_prompt | llm | extract_content
        st.session_state.hyde_chain = hyde_chain
        embedding = HyDEEmbeddings(hyde_chain, base_embeddings)
    else:
        embedding = base_embeddings

    # Vector dimension comes from the base (document) embeddings.
    dim = len(base_embeddings.embed_query("dimension probe"))

    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"rag_collection_{uuid4()}"
    collection_name = st.session_state.collection_name

    # Fresh in-memory client + dense collection.
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )

    # Tag every chunk with a stable id (used by the embedding visualization).
    uuids = [str(uuid4()) for _ in range(len(splits))]
    for split, uid in zip(splits, uuids):
        if not getattr(split, "metadata", None):
            split.metadata = {}
        split.metadata["unique_id"] = uid

    for i in range(0, len(splits), batch_size):
        vectorstore.add_documents(
            documents=splits[i:i + batch_size],
            ids=uuids[i:i + batch_size],
        )

    return vectorstore


class ScoredDenseRetriever(BaseRetriever):
    """Dense Qdrant retriever that records each hit's similarity score in
    ``metadata['similarity_score']`` so the UI can show *why* a chunk ranked.

    Exposes ``vectorstore`` and ``search_kwargs`` so it stays drop-in compatible
    with the hybrid/reranker builders that read those attributes.
    """

    vectorstore: Any
    search_type: str = "similarity"
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        k = self.search_kwargs.get("k", 5)
        if self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=self.search_kwargs.get("fetch_k", 20)
            )
        docs = []
        for doc, score in self.vectorstore.similarity_search_with_score(query, k=k):
            doc.metadata = {**doc.metadata, "similarity_score": round(float(score), 4)}
            docs.append(doc)
        return docs


def get_retriever(vectorstore, search_type="similarity", k=5, fetch_k=20):
    """Pure dense-vector retriever over the Qdrant collection (with scores)."""
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k

    return ScoredDenseRetriever(
        vectorstore=vectorstore,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


# --- Reranking (custom, langchain-core only — no langchain_classic) -----------

def _build_compressor(reranker_type: str, top_n: int) -> Callable[[List[Document], str], List[Document]]:
    """Return a ``compress(docs, query) -> docs`` callable for the chosen reranker."""
    if reranker_type == "cohere":
        cohere_key = _get_cohere_key()
        if not cohere_key:
            raise ValueError(
                "Cohere reranker needs a COHERE_API_KEY (add it to "
                ".streamlit/secrets.toml or the environment), or pick the "
                "'huggingface (BAAI/bge-reranker-base)' reranker instead."
            )
        cohere = CohereRerank(
            model="rerank-english-v3.0",
            top_n=top_n,
            cohere_api_key=cohere_key,
        )

        def _cohere_compress(docs: List[Document], query: str) -> List[Document]:
            reranked = list(cohere.compress_documents(docs, query))
            for d in reranked:  # surface Cohere's relevance score uniformly
                if "relevance_score" in d.metadata:
                    d.metadata["rerank_score"] = round(float(d.metadata["relevance_score"]), 4)
            return reranked

        return _cohere_compress

    elif reranker_type == "huggingface (BAAI/bge-reranker-base)":
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

        def _hf_compress(docs: List[Document], query: str) -> List[Document]:
            if not docs:
                return docs
            scores = model.score([(query, d.page_content) for d in docs])
            ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
            out = []
            for doc, score in ranked[:top_n]:
                doc.metadata = {**doc.metadata, "rerank_score": round(float(score), 4)}
                out.append(doc)
            return out

        return _hf_compress

    raise ValueError("Invalid reranker type. Choose 'cohere' or 'huggingface (BAAI/bge-reranker-base)'.")


class RerankingRetriever(BaseRetriever):
    """Wraps a base retriever and reranks its hits with a cross-encoder/Cohere."""

    base_retriever: BaseRetriever
    compress: Callable[[List[Document], str], List[Document]]

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        return list(self.compress(docs, query))


def get_reranker_retriever(base_retriever, reranker_type="cohere", top_n=3):
    return RerankingRetriever(
        base_retriever=base_retriever,
        compress=_build_compressor(reranker_type, top_n),
    )


# --- Hybrid retrieval (explicit BM25 + dense, weighted RRF fusion) ------------

class WeightedEnsembleRetriever(BaseRetriever):
    """Reciprocal-rank-fusion ensemble of retrievers with per-retriever weights.

    This is the transparent, teachable form of hybrid search: the lexical
    (BM25) and dense (vector) retrievers each return a ranked list, and we fuse
    them with weighted RRF: ``score(d) = Σ_i weight_i / (c + rank_i(d))``.
    """

    retrievers: List[BaseRetriever]
    weights: List[float]
    k: int = 5
    c: int = 60  # RRF dampening constant

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = doc.metadata.get("unique_id") or doc.page_content
                doc_map[key] = doc
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (self.c + rank + 1))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for key, score in ranked[:self.k]:
            doc = doc_map[key]
            doc.metadata = {**doc.metadata, "fusion_score": round(float(score), 4)}
            out.append(doc)
        return out


def get_hybrid_retriever(splits, vector_retriever, bm25_weight=0.5, vector_weight=0.5):
    """Hybrid retriever: BM25 (lexical) + dense vector, fused via weighted RRF."""
    k = vector_retriever.search_kwargs.get("k", 5)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = k

    return WeightedEnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
        k=k,
    )


def get_hybrid_reranker_retriever(splits, vector_retriever, bm25_weight=0.5, vector_weight=0.5, reranker_type="cohere", top_n=3):
    hybrid_retriever = get_hybrid_retriever(splits, vector_retriever, bm25_weight, vector_weight)
    return RerankingRetriever(
        base_retriever=hybrid_retriever,
        compress=_build_compressor(reranker_type, top_n),
    )


def retrieve_documents(retriever, query):
    """Retrieve documents using the given retriever and query."""
    return retriever.invoke(query)


def _scroll_all_points(client: QdrantClient, collection_name: str):
    """Fetch every point (payload + vectors) from a (small, in-memory) collection."""
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=256,
            offset=offset,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def create_embedding_visualization(vectorstore, docs_processed: List[Document], query: str, retrieved_docs: Dict[str, List[Document]], method='pacmap', sample_size=None, dimensions=2, use_random_state=True, selected_retrievers=None, api_key=None):
    """
    Create a visualization of document embeddings and retrieved documents.

    :param vectorstore: The Qdrant vector store containing document embeddings
    :param docs_processed: List of all processed documents
    :param query: The user query
    :param retrieved_docs: Dictionary of retrieved documents for each retriever
    :param method: Dimensionality reduction method ('pacmap', 'umap', or 'tsne')
    :param sample_size: Number of documents to sample (None for all)
    :param dimensions: Number of dimensions for the projection (2 or 3)
    :param use_random_state: Whether to use a fixed random state for reproducibility
    :param selected_retrievers: List of retrievers to visualize
    :param api_key: API key (required for OpenAI)
    :return: Plotly figure object
    """
    provider = st.session_state.get("llm_provider", "Google Gemini")

    # Only require API key for OpenAI
    if provider == "OpenAI" and not api_key:
        raise ValueError("OpenAI API Key is required. Please provide it in the sidebar.")

    # Pull all stored points (vectors + payload) from Qdrant.
    points = _scroll_all_points(vectorstore.client, vectorstore.collection_name)
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []
    documents: List[str] = []
    for point in points:
        vector = point.vector
        if isinstance(vector, dict):  # named vectors -> take the first one
            vector = next(iter(vector.values()))
        payload = point.payload or {}
        embeddings.append(vector)
        metadatas.append(payload.get("metadata", {}) or {})
        documents.append(payload.get("page_content", "") or "")

    # Sample if necessary
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = [embeddings[i] for i in indices]
        metadatas = [metadatas[i] for i in indices]
        documents = [documents[i] for i in indices]

    # Add query embedding using the appropriate provider
    embedding_model = get_embeddings(api_key=api_key, provider=provider)
    query_vector = embedding_model.embed_query(query)
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


# Add new function to display HYDE generations
def display_hyde_generations(query: str, return_doc: bool = False) -> Optional[str]:
    """
    Generate and display HYDE document for a query.

    Args:
        query (str): The query to generate a hypothetical document for
        return_doc (bool): Whether to return the generated document

    Returns:
        Optional[str]: The generated document if return_doc is True, None otherwise
    """
    try:
        # Generate hypothetical document and extract content from AIMessage
        hyde_response = st.session_state.hyde_chain.invoke({"question": query})
        hyde_doc = hyde_response.content if hasattr(hyde_response, 'content') else str(hyde_response)

        with st.expander("View HYDE Generated Document"):
            st.markdown("### Hypothetical Document")
            st.write(hyde_doc)

            # Add some analysis
            st.markdown("### Document Analysis")
            st.markdown(f"- Document length: {len(hyde_doc)} characters")
            st.markdown(f"- Word count: {len(hyde_doc.split())}")

        if return_doc:
            return hyde_doc

    except Exception as e:
        st.error(f"Error generating HYDE document: {str(e)}")
