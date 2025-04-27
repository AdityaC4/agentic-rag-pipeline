from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Uses TavilySearch to perform a web search for the user's question and appends the results to the document list.
    Returns the updated state dict with the new documents.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})
    # print(tavily_results)
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results["results"]]
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {
        "documents": documents,
        "question": question,
    }


if __name__ == "__main__":
    # Quick manual test for the web_search node
    print(web_search({"question": "agent memory", "documents": None}))
