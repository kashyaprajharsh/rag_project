from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import requests
import os
import time
import numpy as np
from collections import Counter
import streamlit as st
from vectordb import get_embeddings


def load_pdf_data(pdf_sources, method='normal'):
    all_data = []
    for pdf_source in pdf_sources:
        try:
            if pdf_source.startswith('http'):
                # For online PDFs, download first then use PyPDFLoader
                response = requests.get(pdf_source)
                response.raise_for_status()  # Raise an error for bad responses
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path, extract_images=(method == 'with_images'))
                pdf_name = pdf_source.split('/')[-1]  # Extract filename from URL

            else:
                # For local files
                if not os.path.exists(pdf_source):
                    raise FileNotFoundError(f"The file {pdf_source} does not exist.")
                
                loader = PyPDFLoader(pdf_source, extract_images=(method == 'with_images'))
                pdf_name = os.path.basename(pdf_source)

            data = loader.load()
            
            # Add a header document for each PDF
            header_doc = {
                "page_content": f"--- Start of PDF: {pdf_name} ---",
                "metadata": {"source": pdf_source, "page": 0}
            }
            all_data.append(header_doc)
            
            # Add the actual PDF data
            all_data.extend(data)
            
            # Add a footer document for each PDF
            footer_doc = {
                "page_content": f"--- End of PDF: {pdf_name} ---",
                "metadata": {"source": pdf_source, "page": len(data) + 1}
            }
            all_data.append(footer_doc)

            # Clean up temporary file if it was created
            if pdf_source.startswith('http'):
                os.remove(tmp_file_path)

        except requests.RequestException as e:
            raise Exception(f"Error downloading the PDF {pdf_source}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading PDF data from {pdf_source}: {str(e)}")

    return all_data

def process_uploaded_files(uploaded_files):
    tmp_file_paths = []
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_paths.append(tmp_file.name)
        return tmp_file_paths
    except Exception as e:
        raise Exception(f"Error processing uploaded files: {str(e)}")

def calculate_content_preservation(original_text, split_chunks):
    # Tokenize and count words in the original text
    original_words = Counter(original_text.lower().split())
    total_original_words = sum(original_words.values())

    # Tokenize and count words in the split chunks
    split_words = Counter(' '.join([chunk.page_content for chunk in split_chunks]).lower().split())

    # Calculate the preservation score
    preserved_count = sum(min(original_words[word], split_words[word]) for word in original_words)
    
    return preserved_count / total_original_words

def calculate_semantic_coherence(chunks, embeddings):
    chunk_embeddings = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    similarities = [np.dot(chunk_embeddings[i], chunk_embeddings[i+1]) / (np.linalg.norm(chunk_embeddings[i]) * np.linalg.norm(chunk_embeddings[i+1])) for i in range(len(chunk_embeddings)-1)]
    return np.mean(similarities)

def calculate_vocabulary_retention(original_text, split_chunks):
    original_vocab = set(original_text.lower().split())
    split_vocab = set(' '.join([chunk.page_content for chunk in split_chunks]).lower().split())
    return len(split_vocab.intersection(original_vocab)) / len(original_vocab)

def split_text(data, splitter_type='character', chunk_size=1000, chunk_overlap=200, semantic_params=None, api_key=None):
    provider = st.session_state.get("llm_provider", "OpenAI")
    
    # Only require API key for semantic splitting with OpenAI
    if splitter_type == 'semantic' and provider == "OpenAI" and not api_key:
        raise ValueError("OpenAI API Key is required for semantic splitting. Please provide it in the sidebar.")
    
    start_time = time.time()
    
    # Filter out header and footer dictionaries
    documents = [doc for doc in data if not isinstance(doc, dict)]
    
    if splitter_type == 'character':
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'recursive':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'token':
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'semantic':
        embeddings = get_embeddings(api_key=api_key, provider=provider)
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=semantic_params['threshold_type'],
            breakpoint_threshold_amount=semantic_params['threshold_amount'],
            number_of_chunks=semantic_params['number_of_chunks']
        )
    else:
        raise ValueError(f"Invalid splitter type: {splitter_type}")

    splits = text_splitter.split_documents(documents)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate metrics
    chunk_sizes = [len(chunk.page_content) for chunk in splits]
    avg_chunk_size = np.mean(chunk_sizes)
    chunk_size_variance = np.var(chunk_sizes)
    coefficient_of_variation = np.std(chunk_sizes) / avg_chunk_size
    chunk_count = len(splits)
    
    original_text = ' '.join([doc.page_content for doc in documents])
    content_preservation = calculate_content_preservation(original_text, splits)
    vocabulary_retention = calculate_vocabulary_retention(original_text, splits)
    
    semantic_coherence = None
    if splitter_type == 'semantic':
        semantic_coherence = calculate_semantic_coherence(splits, embeddings)
    
    return {
        'splits': splits,
        'processing_time': processing_time,
        'avg_chunk_size': avg_chunk_size,
        'chunk_size_variance': chunk_size_variance,
        'coefficient_of_variation': coefficient_of_variation,
        'chunk_count': chunk_count,
        'content_preservation': content_preservation,
        'vocabulary_retention': vocabulary_retention,
        'semantic_coherence': semantic_coherence
    }

