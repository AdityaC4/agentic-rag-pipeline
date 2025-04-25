from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the workflow graph at any point.
    Attributes:
        question: The user question being answered.
        generation: The LLM-generated answer (if available).
        websearch: Whether a web search should be triggered (bool).
        documents: List of retrieved or searched documents.
    """

    question: str
    generation: str
    websearch: bool
    documents: List[str]
