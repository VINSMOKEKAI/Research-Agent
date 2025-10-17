import os 
# IMPORTANT: Use the correct modern import path
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json
from fastapi import FastAPI
from langserve import add_routes
# Pydantic is required for the SecretStr type
from pydantic import SecretStr

from dotenv import load_dotenv
load_dotenv()

# --- OpenRouter Configuration ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "nvidia/nemotron-nano-9b-v2:free" # Example Model

def get_openrouter_llm(model: str = OPENROUTER_MODEL_NAME, temperature: float = 0.7):
    # 1. Retrieve the API key as a standard string
    api_key_str = os.environ.get("OPENROUTER_API_KEY")

    # 🐛 FIX 2: Explicitly raise an error if the key is missing to confirm env setup
    if not api_key_str:
        raise ValueError(
            "The OPENROUTER_API_KEY environment variable is not set or is empty. "
            "Please ensure you run 'export OPENROUTER_API_KEY=\"...\"' before execution."
        )

    # Wrap the valid key string in SecretStr
    api_key_secret = SecretStr(api_key_str)
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        # 🐛 FIX 3: Pass the base_url constant directly. 
        # ChatOpenAI adds the '/chat/completions' part internally.
        base_url=OPENROUTER_BASE_URL, 
        # Correct parameter name for API key and pass the SecretStr object
        api_key=api_key_secret,
    )

# Instantiate the LLMs
try:
    llm_research = get_openrouter_llm(temperature=0.3) 
    llm_search_queries = get_openrouter_llm(temperature=0.0)
    llm_writer = get_openrouter_llm(temperature=0.3)
except ValueError as e:
    # If the key is missing, this will print the explicit message
    print(f"LLM Initialization Error: {e}")
    exit(1) # Exit the script if initialization fails
# Instantiate the LLMs
llm_research = get_openrouter_llm(temperature=0.3) 
llm_search_queries = get_openrouter_llm(temperature=0.0)
llm_writer = get_openrouter_llm(temperature=0.3)



RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text} 

-----------

Using the above text, answer in short the following question: 

> {question}

-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def scrape_text(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"


scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
) | SUMMARY_PROMPT | llm_research | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | llm_search_queries | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------

Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.

You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | llm_writer | StrOutputParser()


# --- FastAPI and LangServe ---

app = FastAPI(
  title="OpenRouter Research Assistant Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces with OpenRouter models",
)

add_routes(
    app,
    chain,
    path="/research-assistant",
)


if __name__ == "__main__":
    import uvicorn
    # Make sure to run the server with the OPENROUTER_API_KEY environment variable set
    uvicorn.run(app, host="localhost", port=8000)