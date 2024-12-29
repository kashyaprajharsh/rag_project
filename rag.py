from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
import time
from langchain_community.callbacks.manager import get_openai_callback
import streamlit as st

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

def get_llm(model_name="gpt-4o-mini", api_key=None,provider="Google Gemini"):
    """Get LLM based on provider"""
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAI models. Please provide it in the sidebar.")
        return ChatOpenAI(model_name=model_name, temperature=0.1, openai_api_key=api_key)
    else:  # Google Gemini
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)

RAG_TYPES = {
    "Vector Store Retriever": "base_retriever",
    "Vector Store Retriever with Reranker": "reranker_retriever",
    "Ensemble Retriever (BM25 + Vector Store)": "hybrid_retriever",
    "Ensemble Retriever with Reranker": "hybrid_reranker_retriever"
}

def generate_rag_answer(prompt, retriever, rag_type, api_key):
    provider = st.session_state.get("llm_provider", "OpenAI")
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