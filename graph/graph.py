from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from graph.const import *
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState


def decide_to_generate(state):
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


workflow = StateGraph(GraphState)

# Add nodes for each major workflow step
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

# Set the entry point of the workflow
workflow.set_entry_point(RETRIEVE)

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
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# Compile the workflow into an executable app
app = workflow.compile()

# Generate a visualization of the workflow as a PNG
app.get_graph().draw_mermaid_png(
    output_file_path="graph.png", draw_method=MermaidDrawMethod.PYPPETEER
)
