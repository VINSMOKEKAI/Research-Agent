from langgraph.graph import StateGraph, END
from src.state.state import ResearchState 
from src.graph.node import call_tool,generate_response
from typing import TypedDict, Dict, Any, List

# Define the LangGraph workflow
workflow = StateGraph(ResearchState)

# 1. Add the nodes
workflow.add_node("search", call_tool)
workflow.add_node("answer", generate_response)

# 2. Set the entry point
workflow.set_entry_point("search")

# 3. Define the edges (the flow between nodes)
workflow.add_edge("search", "answer")

# 4. Define the final step
workflow.add_edge("answer", END)

# Compile the graph
research_agent_graph = workflow.compile()

# Print the graph schema for debugging (optional, but helpful)
print("--- Research Agent Graph Compiled ---")
# print(research_agent_graph.get_graph().draw_ascii()) # Uncomment to see ASCII graph