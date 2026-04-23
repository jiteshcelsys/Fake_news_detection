import os
from typing import Annotated, List
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class FakeNewsResult(BaseModel):
    label: str = Field(description="Return ONLY 'yes' if fake news, 'no' if real news")
    reason: str = Field(description="Short explanation of why the news is fake or real")
    evidence_links: List[str] = Field(description="List of URLs used to verify the claim")


class AgentState(TypedDict):
    news_input: str
    messages: Annotated[list[BaseMessage], add_messages]  # proper reducer so ToolNode appends correctly
    result: FakeNewsResult | None


SYSTEM_PROMPT = """You are a professional fact-checking agent. Your job is to verify news claims.

Steps you MUST follow:
1. Extract the key factual claims from the news input
2. Use the search tool to find evidence for or against those claims
3. Search at least 2-3 times with different queries to gather enough evidence
4. Based on retrieved evidence, classify the news

Classification rules:
- Return label="yes" if the news is FAKE or unverifiable
- Return label="no" if the news is REAL and confirmed by credible sources
- NEVER return anything other than "yes" or "no" for label
- ALWAYS include at least 1-3 evidence_links from your search results
- If no strong evidence found, return label="yes" (assume suspicious)

IMPORTANT: You must call the search tool before making your final classification.
Do NOT hallucinate links. Only use URLs from actual search results."""


def create_agent():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        api_key=api_key,
    )

    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm.with_structured_output(FakeNewsResult)

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "finalize"

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}  # add_messages reducer appends this

    def finalize_node(state: AgentState):
        news_input = state["news_input"]
        messages = state["messages"]

        tool_results = [
            msg.content for msg in messages
            if hasattr(msg, "content") and isinstance(msg.content, str)
            and getattr(msg, "type", "") == "tool"
        ]
        context = "\n\n".join(tool_results) if tool_results else "No search results available."

        finalize_prompt = f"""Based on your research, classify this news claim:

NEWS: {news_input}

SEARCH EVIDENCE GATHERED:
{context}

Provide your final structured classification:
- label: "yes" = fake/unverifiable, "no" = real/confirmed
- Include actual URLs found in search results as evidence_links
- Keep reason concise (2-4 lines)"""

        result = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=finalize_prompt),
        ])
        return {"result": result}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()


def check_news(news_input: str) -> FakeNewsResult:
    agent = create_agent()

    state = agent.invoke({
        "news_input": news_input,
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Fact-check this news claim: {news_input}"),
        ],
        "result": None,
    })

    return state["result"]


def main():
    print("=== Fake News Detection Agent ===\n")

    test_cases = [
        "Government announces free electricity for all citizens starting tomorrow.",
        "NASA confirms water ice discovered on the Moon's south pole.",
    ]

    for news in test_cases:
        print(f"NEWS: {news}")
        print("-" * 60)
        result = check_news(news)
        print(f"LABEL:  {'FAKE' if result.label == 'yes' else 'REAL'} ({result.label})")
        print(f"REASON: {result.reason}")
        print("EVIDENCE:")
        for link in result.evidence_links:
            print(f"  - {link}")
        print("\n")


if __name__ == "__main__":
    main()
