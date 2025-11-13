from langchain_community.tools import DuckDuckGoSearchRun
from typing import List

duckduckgo_search_tool = DuckDuckGoSearchRun(
    name="DuckDuckGo_Search",
    description=(
        "A tool for **searching the web** for **up-to-date information, news, "
        "and research articles** on a given topic or query. Use this tool "
        "when you need external, current data."
    )
)

tools: List[DuckDuckGoSearchRun] = [duckduckgo_search_tool]