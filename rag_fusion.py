from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import json
from json import dumps, loads
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Dict, Any
from langchain.schema import Document

def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[tuple]:
    """
    Implements the Reciprocal Rank Fusion algorithm for combining multiple ranked lists.
    
    Args:
        results: List of lists containing ranked documents
        k: Constant in RRF formula (default: 60)
    
    Returns:
        List of tuples (document, score) sorted by fusion score
    """
    fused_scores: Dict[str, float] = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            # Create a unique key for each document using its ID
            doc_key = doc.metadata.get('unique_id', str(hash(doc.page_content)))
            
            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0
            
            # Apply RRF formula
            fused_scores[doc_key] += 1 / (rank + k)
    
    # Sort documents by fusion score
    reranked_docs = []
    for doc_key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        # Find original document
        for doc_list in results:
            for doc in doc_list:
                if doc.metadata.get('unique_id', str(hash(doc.page_content))) == doc_key:
                    reranked_docs.append((doc, score))
                    break
            if len(reranked_docs) > 0 and reranked_docs[-1][0].metadata.get('unique_id') == doc_key:
                break
    
    return reranked_docs

def generate_fusion_queries(question: str, llm: Any) -> List[str]:
    """Generate multiple search queries for a single input query."""
    fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query.
    Generate 4 different but related search queries for: {question}
    
    The queries should:
    - Explore different aspects of the question
    - Use different phrasings
    - Focus on key concepts
    - Be clear and specific
    
    Output (4 queries):"""
    
    fusion_prompt = PromptTemplate(
        template=fusion_template,
        input_variables=["question"]
    )
    
    chain = fusion_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    queries = [q.strip() for q in chain.invoke({"question": question}) if q.strip()]
    return queries[-4:]  # Ensure we get exactly 4 queries

def create_fusion_visualization(original_results: List[Document], 
                              fused_results: List[tuple],
                              queries: List[str]) -> Dict[str, Any]:
    """
    Create visualization data for RAG Fusion results.
    
    Returns:
        Dictionary containing plot data and metrics
    """
    # Prepare data for visualization
    original_ranks = {doc.metadata.get('unique_id'): i 
                     for i, doc in enumerate(original_results)}
    
    fused_ranks = {doc.metadata.get('unique_id'): i 
                  for i, (doc, _) in enumerate(fused_results)}
    
    # Calculate metrics
    metrics = {
        "num_original_docs": len(original_results),
        "num_fused_docs": len(fused_results),
        "queries_generated": len(queries),
        "rank_changes": {}
    }
    
    # Track rank changes
    for doc_id, orig_rank in original_ranks.items():
        if doc_id in fused_ranks:
            metrics["rank_changes"][doc_id] = {
                "original_rank": orig_rank,
                "fused_rank": fused_ranks[doc_id],
                "change": orig_rank - fused_ranks[doc_id]
            }
    
    return {
        "metrics": metrics,
        "original_ranks": original_ranks,
        "fused_ranks": fused_ranks,
        "fused_scores": [score for _, score in fused_results]
    }

def get_fusion_chain(retriever: Any, llm: Any):
    """Create a RAG Fusion chain combining query generation and retrieval."""
    def process_queries(input_dict: Dict[str, str]) -> List[List[Document]]:
        queries = generate_fusion_queries(input_dict["question"], llm)
        return [retriever.get_relevant_documents(q) for q in queries]
    
    return {
        "queries": lambda x: generate_fusion_queries(x["question"], llm),
        "retrieved_docs": process_queries,
        "fused_docs": lambda x: reciprocal_rank_fusion(x["retrieved_docs"])
    }
