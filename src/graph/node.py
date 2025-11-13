import os
from typing import cast, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from src.state.state import ResearchState # Assuming this is the 'ResearchState' I fixed earlier
from src.tools.tools import duckduckgo_search_tool
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
# LLM 
llm = ChatOllama(model="llama3:latest")


# --- Nodes ---

def call_tool(state: ResearchState) -> Dict[str, Any]:
    """
    Node: Call the search tool based on the user query and update the state.
    """
    search_query = f"Latest news, articles, and research papers about {state['query']}"

    try:
        result = duckduckgo_search_tool.invoke(search_query)
    except Exception as e:
        print(f"Error calling search tool: {e}")
        result = "No search results found due to an error."
    return {"search_results": result}


def generate_response(state: ResearchState) -> Dict[str, str]:
    """
    Node: Uses the LLM to process search results/history and generate the final response.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "You are a high-precision Research Agent designed to gather, analyze, and summarize information with maximum accuracy and speed. "
                "Your role is to: Understand the user’s query, analyze the provided search results, and generate a final, structured response. "
                "Follow these rules strictly:\n"
                "- Always cite sources when available.\n"
                "- If data is uncertain, explicitly state uncertainty.\n"
                "- Never guess or fabricate statistics.\n"
                "- Prioritize recency, authoritative sources, and verifiable facts.\n"
                "- **Provide final answers in this structure**:\n"
                "    1. **Summary** (bullet points, short)\n"
                "    2. **Key Findings**\n"
                "    3. **Sources** (Cite the text snippets provided below)\n\n"
                "Tone: Professional, analytical, and direct. Focus on accuracy, clarity, and relevance."
            ),
            # 4. FIX: Use the correct state key 'search_results' and 'query'
            HumanMessage(
                content=f"User Query: {state['query']}\n\n"
                        f"--- Web Search Results for latest query ---\n"
                        f"{state['search_results']}" # This holds the search results string
            )
        ]
    )

    chain = prompt | llm
    response = chain.invoke(cast(Dict[str, Any], state))
    return {"final_answer": response.content} 