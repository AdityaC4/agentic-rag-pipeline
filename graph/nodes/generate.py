from typing import Any, Dict

from graph.state import GraphState


from graph.chains.generation import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    context = state["documents"]

    generation = generation_chain.invoke({"context": context, "question": question})
    return {"generation": generation, "question": question, "documents": context}
