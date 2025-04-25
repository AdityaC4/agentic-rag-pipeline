from typing import Any, Dict

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from langchain.schema import Document
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grades each document for relevance to the user question using an LLM grader.
    Returns an updated state dict with filtered documents and a flag for websearch.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated websearch state
    """

    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        grade: GradeDocuments = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if grade.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
    return {
        "documents": filtered_docs,
        "question": question,
        "websearch": web_search,
    }
