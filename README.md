🧠 LangGraph Research AgentThis project implements a high-precision, autonomous research agent using LangGraph to manage complex workflows, LangChain for component integration, and Ollama for large language model (LLM) execution.The agent takes a user query, performs a web search using DuckDuckGo, and synthesizes the findings into a structured, final report.✨ FeaturesModular Architecture: Uses LangGraph's StateGraph to define a clear, sequential flow.Real-Time Grounding: Employs DuckDuckGoSearchRun for up-to-date, real-time information retrieval.Ollama Integration: Utilizes the llama3:latest model for query processing and final summary generation.Structured Output: LLM is guided by a System Prompt to deliver professional, analytical, and structured findings.⚙️ PrerequisitesBefore running the agent, you must have the following installed:Python 3.9+Ollama: The Ollama service must be running locally.# Install the specific model used in the project
ollama pull llama3:latest
⬇️ InstallationClone the repository (or set up your files):git clone <https://github.com/VINSMOKEKAI/Research-Agent.git>
cd Research-Agent
Create and activate a virtual environment (optional but recommended):python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows
Install Python dependencies:The project relies on the following packages:pip install langchain langchain-core langchain-community langgraph langchain-ollama duckduckgo-search
🚀 UsageRun the agent directly from your terminal by providing a research query in quotes as a command-line argument to main.py.python main.py "What are the key differences between LangGraph and Autogen?"
Example Output🧠 Initializing Research Agent for topic: 'What are the key differences between LangGraph and Autogen?'
--- Research Agent Graph Compiled ---

================================================================================
✅ RESEARCH COMPLETE
================================================================================
<LLM-Generated Summary and Findings based on search results will appear here>
================================================================================
📂 Project StructureThe workflow is organized into modular files:.
├── src/
│   ├── state/
│   │   └── state.py      # Defines the ResearchState (TypedDict) for the graph
│   ├── tools/
│   │   └── tools.py      # Initializes the DuckDuckGoSearchRun tool
│   └── graph/
|       └── graph.py      # Defines and compiles the LangGraph StateGraph (The workflow logic)
│       └── nodes.py      # Contains call_tool (search) and generate_response (LLM) functions     
└── main.py               # Entry point, handles CLI arguments and executes the graph

⚙️ Workflow BreakdownThe agent follows a simple, two-step sequential workflow defined in research_graph.py:search Node (call_tool function):Reads the query from the state.Executes the DuckDuckGo_Search tool with a refined query.Updates the state with the search_results.answer Node (generate_response function):Receives the state containing the original query and the search_results.Passes both to the llama3:latest LLM using a structured prompt.The LLM analyzes the results and generates a final summary.Updates the state with the final_answer.END: The graph terminates, and the result is printed by main.py.