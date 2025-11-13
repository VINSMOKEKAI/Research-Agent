from typing import TypedDict, List, Annotated
from langgraph.graph.message import AnyMessage

class ResearchState(TypedDict):
    """
    Represents the state of our research agent.

    This is defined as a TypedDict, which is immutable, so methods
    like 'add_messages' are used on the 'messages' field if it were used.
    For this simple state, we just define the fields.
    """
    query: str
    search_query: str
    search_results: List[str]
    final_answer: str
