import sys
import os
from src.graph.graph import research_agent_graph
from src.state.state import ResearchState # Assuming ResearchState is in this path
from typing import Dict, Any

# A function to run the agent
def run_agent(topic: str):
    """
    Runs the compiled LangGraph agent with an initial query.
    """
    print(f"\n🧠 Initializing Research Agent for topic: '{topic}'")
    
    # 1. Define the initial state for the graph
    # FIX: Initialize 'search_results' as an empty list [] 
    # to match the expected List[str] type from ResearchState.
    initial_state: ResearchState = {
        "query": topic,
        "search_query": "",
        "search_results": [], # CHANGED from "" to []
        "final_answer": ""
    }
    
    # 2. Configure the graph (use synchronous invoke for simplicity)
    try:
        # Note: LangGraph's invoke expects a standard dictionary for state updates
        result = research_agent_graph.invoke(initial_state)

        # 3. Extract and display the final answer
        final_state: Dict[str, Any] = result 
        final_answer = final_state.get("final_answer", "Agent did not produce a final answer.")

        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETE")
        print("="*80)
        print(final_answer)
        print("="*80)
        
        # Optional: show raw search data for verification
        # print("\n--- Raw Search Results Used ---")
        # print(final_state.get("search_results"))

    except Exception as e:
        print(f"\n❌ An error occurred during graph execution: {e}")
        print("Please check your node functions or LLM configuration.")

def main():
    """
    Main entry point for terminal execution.
    """
    # Simple check for command-line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<Your research query here>\"")
        print("Example: python main.py \"The latest developments in quantum computing in 2025\"")
        sys.exit(1)
    
    # Get the topic from the command line
    topic = " ".join(sys.argv[1:])
    
    # Run the agent
    run_agent(topic)

if __name__ == "__main__":
    main()