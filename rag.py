from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
import time
from langchain_community.callbacks.manager import get_openai_callback

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """You are an Expert DATA ANALYST and INFORMATION INTERPRETER. Your task is to EXTRACT and PROVIDE answers to the user's questions using the given context.

Follow these steps:

1. THOROUGHLY READ the provided context labeled context :"{context}" to fully grasp the information contained within.

2. ANALYZE the user's question labeled "{question}" and CROSS-REFERENCE it with the context.


3. If you find the ANSWER within the context, RESPOND with a CLEAR and DIRECT answer, citing SPECIFIC PARTS of the context when applicable.

4. In case the CONTEXT LACKS the necessary information to answer, CONFIDENTLY STATE "I don't have enough information to answer that question."

5. ENSURE your response is CONCISE and PRECISE, avoiding any external knowledge or assumptions not present in the given context.

Remember, I’m going to tip $300K for a BETTER SOLUTION!

Now Take a Deep Breath.

Your response:"""

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

def get_llm(model_name="gpt-4o-mini", api_key=None):
    if not api_key:
        raise ValueError("OpenAI API Key is required. Please provide it in the sidebar.")
    return ChatOpenAI(model_name=model_name, temperature=0.1, openai_api_key=api_key)

RAG_TYPES = {
    "Naive RAG": "base_retriever",
    "RAG with Reranker": "reranker_retriever",
    "RAG with Hybrid Search": "hybrid_retriever",
    "Advanced RAG": "hybrid_reranker_retriever"
}

def generate_rag_answer(prompt, retriever, rag_type, api_key):
    llm = get_llm(api_key=api_key)
    chain = create_rag_chain(retriever, llm)
    question = prompt['question'] if isinstance(prompt, dict) else prompt
    
    start_time = time.time()
    with get_openai_callback() as cb:
        result = chain.invoke(question)
        end_time = time.time()
        #print(cb)
    
    total_duration = end_time - start_time
    
    # Extract the prompt from the custom_rag_prompt
    prompt_template = custom_rag_prompt.template.format(context=format_docs(result["context"]), question=question)
    
    return {
        "input": question,
        "output": result["answer"],
        "context": result["context"],
        "prompt": prompt_template,  # Add the prompt to the returned results
        "steps": [
            {"name": "Retriever", "duration": total_duration / 2},  # Estimate
            {"name": "LLM", "duration": total_duration / 2}  # Estimate
        ],
        "startTime": start_time * 1000,  # Convert to milliseconds
        "endTime": end_time * 1000,
        "status": "Success",
        "totalTokens": cb.total_tokens,
        "promptTokens": cb.prompt_tokens,
        "completionTokens": cb.completion_tokens,
        "cost": cb.total_cost,
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