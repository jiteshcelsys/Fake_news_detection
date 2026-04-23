import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search.invoke("Obama's first name?")
from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults(backend="news",output_format="list")

# TASK - 1 [Tools + Model]

search = DuckDuckGoSearchRun()
model = ChatAnthropic(model="claude-opus-4-7")

# Bind search tool so model can call it
model_with_search = model.bind_tools([search])
## TASK - 2[Schema] — Define a structured output schema using Pydantic for the fact-checking results.

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal

class FakeNewsResult(BaseModel):
    label: Literal["yes", "no"] = Field(
        description="Return 'yes' if FAKE, 'no' if REAL"
    )
    reason: str = Field(
        description="Concise explanation (2–4 lines)"
    )
    evidence_links: List[str] = Field(
        description="List of 1–3 credible URLs (NO Wikipedia)"
    )

    @field_validator("evidence_links")
    @classmethod
    def remove_wikipedia(cls, links):
        filtered = [l for l in links if "wikipedia.org" not in l.lower()]
        if not filtered:
            raise ValueError("Only Wikipedia links found. Need credible sources.")
        return filtered
    
def analyze_news(news_statement: str) -> FakeNewsResult:
    system_msg = """
You are a strict fact-checking assistant.

Rules:
- ALWAYS use the search tool
- NEVER rely on prior knowledge
- DO NOT include Wikipedia links
- Use at least 2 sources
"""

    messages = [
        {"role": "system", "content": system_msg},
        HumanMessage(content=f"Verify this news:\n\n{news_statement}")
    ]

    response = model_with_search.invoke(messages)

    # 🔁 Ensure search happens
    if not response.tool_calls:
        # force search manually
        search_result = search.invoke(news_statement)
        messages.append(ToolMessage(content=search_result, tool_call_id="manual"))
    else:
        messages.append(response)
        for tool_call in response.tool_calls:
            search_result = search.invoke(tool_call["args"]["query"])
            messages.append(
                ToolMessage(content=search_result, tool_call_id=tool_call["id"])
            )

    # 🎯 Final structured output
    final_model = model.with_structured_output(FakeNewsResult)

    messages.append({
        "role": "user",
        "content": "Classify as FAKE or REAL with valid sources (no Wikipedia)."
    })

    return final_model.invoke(messages)

result = analyze_news(
    "Arivnd Modi has bought Twitter and renamed it to X"
)
result
