from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from graph.const import *
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import question_router, RouteQuery


def decide_to_generate(state: GraphState):
    """
    Decide whether to generate an answer or trigger a web search based on document grading.
    Returns the next node name for the workflow.
    """
    print("---ASSESS GRADED DOCS---")

    if state["websearch"]:
        print("DECISION: NOT ALL DOCS WERE RELEVANT")
        return WEB_SEARCH
    else:
        print("DECISION: ALL DOCS WERE RELEVANT")
        return GENERATE


def grade_generation_grounded_in_docs_and_question(state: GraphState):
    """
    Grade the LLM generation for relevance to the question and grounding in the retrieved documents.
    Returns the next node name for the workflow.
    """
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-SEARCH---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.data_source == "vectorstore":
        print("---DECISION: VECTORSTORE---")
        return RETRIEVE
    elif source.data_source == "websearch":
        print("---DECISION: WEBSEARCH---")
        return WEB_SEARCH
    else:
        raise ValueError(f"Unknown data source: {source.data_source}")


workflow = StateGraph(GraphState)

# Add nodes for each major workflow step
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

# Set the entry point of the workflow
workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEB_SEARCH: WEB_SEARCH,
    },
)

# Define the workflow edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_docs_and_question,
    {
        "useful": END,
        "not useful": WEB_SEARCH,
        "not supported": GENERATE,
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# Compile the workflow into an executable app
app = workflow.compile()

# Generate a visualization of the workflow as a PNG
app.get_graph().draw_mermaid_png(
    output_file_path="graph.png", draw_method=MermaidDrawMethod.PYPPETEER
)
