from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved doucments are relevatn to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---GRADE DOCUMENTS---")
    documents = state["documents"]
    question = state["question"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        grade = retrieval_grader.invoke(
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
        "web_search": web_search,
    }
