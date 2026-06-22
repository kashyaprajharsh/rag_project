from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
import time

# Default model per canonical provider; any model_name can be passed to get_llm().
DEFAULT_MODELS = {
    "google_genai": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
}


def _canonical_provider(provider: str) -> str:
    """Map the app's display names to LangChain provider ids."""
    p = (provider or "").strip().lower()
    if p in ("openai", "gpt"):
        return "openai"
    if p in ("anthropic", "claude"):
        return "anthropic"
    return "google_genai"  # default: Google Gemini
from langchain_community.callbacks.manager import get_openai_callback
import streamlit as st

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_content(response) -> str:
    """Flatten a chat-model response to plain text.

    Gemini (and other reasoning models) may return ``content`` as a list of
    blocks like ``[{"type": "text", "text": "..."}, {...thinking...}]`` instead
    of a string. Pull out just the text. Also accepts a raw string/list.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return content or ""


template = """Based on the following context, answer the question: {question}            
Context:
{context}

Answer:"""
# template = """You are a helpful AI assistant. Use the following pieces of context to answer the user's question. If the answer cannot be found in the context, respond with "I don't have enough information to answer that question."

# Context:
# {context}

# User's Question: {question}

# Instructions:
# 1. Carefully analyze the context provided.
# 2. Answer the question based solely on the information in the context.
# 3. If the context doesn't contain the answer, say so.
# 4. Provide a clear, concise, and accurate response.
# 5. If relevant, cite specific parts of the context to support your answer.
# 6. Do not use any external knowledge or make assumptions beyond the given context.

# Your response:"""

custom_rag_prompt = PromptTemplate.from_template(template)

def create_rag_chain(retriever, llm):
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | extract_content  # robust to Gemini list-style content blocks
    )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

def get_llm(model_name=None, api_key=None, provider="Google Gemini", temperature=0.1):
    """Initialize a chat model via LangChain's ``init_chat_model``.

    Any model can be passed; ``provider`` accepts the app's display names
    ('OpenAI', 'Google Gemini') or canonical ids ('openai', 'google_genai',
    'anthropic'). If ``model_name`` is omitted, a sensible per-provider default
    is used. Mirrors the multi-provider pattern in the antarman project.
    """
    prov = _canonical_provider(provider)
    # Fall back to the model the user typed in the sidebar, then the provider default.
    if model_name is None:
        try:
            model_name = st.session_state.get("model_name")
        except Exception:
            model_name = None
    model = model_name or DEFAULT_MODELS[prov]

    if prov == "openai":
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAI models. Please provide it in the sidebar.")
        return init_chat_model(model, model_provider="openai", temperature=temperature, api_key=api_key)

    if prov == "google_genai":
        # Use ChatGoogleGenerativeAI directly — init_chat_model doesn't pass
        # Gemini streaming params through cleanly (per antarman's llm.py).
        gemini_kwargs = {"model": model, "temperature": temperature}
        if api_key:
            gemini_kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**gemini_kwargs)

    # anthropic / any other init_chat_model-supported provider
    return init_chat_model(model, model_provider=prov, temperature=temperature)

RAG_TYPES = {
    "Vector Store Retriever": "base_retriever",
    "Vector Store Retriever with Reranker": "reranker_retriever",
    "Ensemble Retriever (BM25 + Vector Store)": "hybrid_retriever",
    "Ensemble Retriever with Reranker": "hybrid_reranker_retriever"
}

def generate_rag_answer(prompt, retriever, rag_type, api_key):
    provider = st.session_state.get("llm_provider", "Google Gemini")
    llm = get_llm(api_key=api_key, provider=provider)
    chain = create_rag_chain(retriever, llm)
    question = prompt['question'] if isinstance(prompt, dict) else prompt
    
    start_time = time.time()
    
    # Only use OpenAI callback for OpenAI provider
    if provider == "OpenAI":
        with get_openai_callback() as cb:
            result = chain.invoke(question)
            end_time = time.time()
            total_tokens = cb.total_tokens
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            total_cost = cb.total_cost
    else:
        result = chain.invoke(question)
        end_time = time.time()
        total_tokens = 0  # Gemini doesn't provide token counts
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0
    
    total_duration = end_time - start_time
    
    # Extract the prompt from the custom_rag_prompt
    prompt_template = custom_rag_prompt.template.format(context=format_docs(result["context"]), question=question)
    
    return {
        "input": question,
        "output": result["answer"],
        "context": result["context"],
        "prompt": prompt_template,
        "steps": [
            {"name": "Retriever", "duration": total_duration / 2},
            {"name": "LLM", "duration": total_duration / 2}
        ],
        "startTime": start_time * 1000,
        "endTime": end_time * 1000,
        "status": "Success",
        "totalTokens": total_tokens,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "cost": total_cost,
        "latency": total_duration
    }

def generate_all_rag_answers(prompt, retrievers, api_key):
    results = {}
    for rag_type, retriever_key in RAG_TYPES.items():
        if retriever_key in retrievers and retrievers[retriever_key] is not None:
            results[rag_type] = generate_rag_answer(prompt, retrievers[retriever_key], rag_type, api_key)
    return results

def get_rag_chain(rag_type, retrievers, api_key):
    retriever_key = RAG_TYPES.get(rag_type)
    if retriever_key and retriever_key in retrievers and retrievers[retriever_key] is not None:
        return create_rag_chain(retrievers[retriever_key], get_llm(api_key=api_key))
    else:
        raise ValueError(f"Invalid RAG type or retriever not available: {rag_type}")