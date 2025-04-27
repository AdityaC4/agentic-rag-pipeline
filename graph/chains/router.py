from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Router a user query to the most relevant datasource."""

    data_source: Literal["websearch", "vectorstore"] = Field(
        ...,
        description="Given a user question choose to router it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are a expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
