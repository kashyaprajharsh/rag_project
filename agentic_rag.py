import os
from typing import List, Dict, Literal
from cohere import SystemMessage
from sympy import assuming
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display
from pprint import pprint
import streamlit as st
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Import necessary components from vectordb.py, rag.py, and pdf.py
from vectordb import (
    get_retriever,
    get_reranker_retriever,
    get_hybrid_retriever,
    get_hybrid_reranker_retriever
)
from rag import (
    create_rag_chain,
    get_llm,
    format_docs,
    custom_rag_prompt
)

# Set environment variables
os.environ['TAVILY_API_KEY'] = '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'
tavily_api_key = os.getenv("TAVILY_API_KEY")

web_search_tool = TavilySearchResults(k=5)

# Data model
class RouteQuery(BaseModel):
    """ Route a user query to the most relevant datasource. """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

### Hallucination Grader 
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
    explanation: str = Field(description="Explain the reasoning for the score")

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, must be exactly 'yes' or 'no'")
    explanation: str = Field(description="Explain the reasoning in a single line without line breaks")

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    web_search : str # Binary decision to run web search
    max_retries : int # Max number of retries for answer generation 
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add] 
    documents : List[str] # List of retrieved documents


# System prompts
ROUTER_SYSTEM_PROMPT  ="""You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks,llm, and large language models,ml, and llm agents.
                                    
Use the vectorstore for questions on these topics. For all else, use web-search."""

# Doc grader instructions 
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

#  Hallucination grader instructions 
hallucination_grader_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


# Answer grader instructions 
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION

(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Score:

A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = "QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}"

# Grader prompt
hallucination_grader_prompt = "FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}"

# Grader prompt
doc_grader_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"



### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}



def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    loop_step = state.get("loop_step", 0)
    
    # Use the existing RAG chain with the retriever
    rag_chain = create_rag_chain(retriever=retriever, llm=llm)
    result = rag_chain.invoke(question)
    
    # Return the generation as a string directly
    return {"generation": result["answer"], "loop_step": loop_step+1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No" 
    for d in documents:
        print(f"\nProcessing document: {d.page_content[:100]}...")  # Show first 100 chars of document
        
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        print(f"\nPrompt sent to LLM: {doc_grader_prompt_formatted}")
        
        score = structured_llm_doc_grader.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        print(f"\nLLM Response (score): {score}")  # See what the LLM is returning
        
        grade = score.binary_score  # This is where the error occurs
        print(f"\nExtracted grade: {grade}")
        
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}
    

def transform_query(state: GraphState) -> GraphState:
    """Transform the query to produce a better question"""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", [])

    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question
    }

    
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


### Edges

def route_question(state):
    """
    Route question to web search or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    source = structured_llm_router.invoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + [HumanMessage(content=state["question"])]) 
    if source.datasource == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3) # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), 
        generation=generation
    )
    score = structured_llm_hallucination_grader.invoke([SystemMessage(content=hallucination_grader_instructions)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        
        # Format the prompt and ensure single-line response
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, 
            generation=generation
        ).replace("\n", " ")
        
        # Add explicit instruction for single-line response
        modified_instructions = answer_grader_instructions + "\nIMPORTANT: Provide your explanation without any line breaks."
        
        score = structured_llm_answer_grader.invoke([
            SystemMessage(content=modified_instructions),
            HumanMessage(content=answer_grader_prompt_formatted)
        ])
        
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"  
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries" 

def create_workflow():
    """Create and configure the workflow graph"""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search) # web search
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("generate", generate) # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "transform_query")
    workflow.add_conditional_edges(
        "transform_query",
        route_question,
        {  
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("websearch", "generate")
    # workflow.add_edge("transform_query","retrieve",state_key="transform_query_retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {  
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )

    
    return workflow.compile()





def initialize_components(api_key=None):
    """Initialize all necessary components"""
    global llm, retriever, rag_chain, question_router, retrieval_grader
    global hallucination_grader, answer_grader, question_rewriter
    global structured_llm_router, structured_llm_doc_grader, structured_llm_hallucination_grader, structured_llm_answer_grader
    
    # Get provider from session state, default to Gemini
    provider = st.session_state.get("llm_provider", "Google Gemini")
    
    # Initialize LLM with provider
    llm = get_llm(api_key=api_key, provider=provider)
    
    # Move structured output initialization here
    structured_llm_router = llm.with_structured_output(RouteQuery)
    structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)
    structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
    structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
    
    # Get the selected retriever from session state
    retriever = st.session_state.get("selected_retriever")
    if retriever is None:
        st.warning("No retriever selected. Please select a retriever in the Retriever page first.")
        return
    
    # Initialize chains
    rag_chain = create_rag_chain(retriever, llm)
 
    
    # Initialize question rewriter
    question_rewriter = ChatPromptTemplate.from_messages([
        ("system", """You are a question re-writer that converts an input question to a better version that is optimized
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning"""),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]) | llm | StrOutputParser()

def run_rag_bot_stream(query: str, api_key=None):
    """Run the RAG bot with streaming output"""
    # Initialize components
    initialize_components(api_key)
    
    # Create workflow
    app = create_workflow()
    
    # # Save the graph as PNG
    # graph_image = app.get_graph().draw_mermaid_png(
    #     draw_method=MermaidDrawMethod.API,
    # )
    
    # # Display the graph
    # display(Image(graph_image))
    
    # # Create directory if it doesn't exist
    # os.makedirs("workflow_graphs", exist_ok=True)
    
    # # Save the image to a file in the workflow_graphs directory
    # with open("workflow_graphs/workflow_graph.png", "wb") as f:
    #     f.write(graph_image)
    
    # Prepare inputs
    inputs = {
        "question": query,
        "documents": [],
        "generation": "",
        "web_search": "No"
    }
    
    # Run with streaming
    config = RunnableConfig(recursion_limit=10)
    for output in app.stream(inputs, config):
        for key, value in output.items():
            #print(f"Finished running: {key}:")
            if isinstance(value, dict) and "generation" in value:
                yield {"type": "generation", "key": key, "content": value["generation"]}
                #pprint(value["generation"])  
            else:
                yield {"type": "process", "key": key, "content": value}
                #pprint(value)
        print("\n")

if __name__ == "__main__":
    # Example usage
    query = "explain sampling in LLM?"
    run_rag_bot_stream(query)
 
   