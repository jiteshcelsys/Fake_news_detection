import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import field_validator
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class CheckRequest(BaseModel):
    input: str  # either raw text or a URL


class FakeNewsResult(BaseModel):
    label: Literal["yes", "no"] = Field(description="'yes' = FAKE, 'no' = REAL")
    reason: str = Field(description="Concise explanation (2-4 lines)")
    evidence_links: List[str] = Field(description="1-3 credible source URLs")
    input_type: Literal["text", "url"] = Field(description="What was analysed")

    @field_validator("evidence_links")
    @classmethod
    def remove_wikipedia(cls, links):
        filtered = [l for l in links if "wikipedia.org" not in l.lower()]
        return filtered or links  # fall back to all links if only wiki found


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"^https?://", re.I)


def is_url(value: str) -> bool:
    return bool(URL_RE.match(value.strip()))


def fetch_article_text(url: str, max_chars: int = 4000) -> str:
    """Scrape visible text from a URL."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as e:
        return f"[Could not fetch article: {e}]"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=api_key)


SYSTEM_TEXT = """You are a strict fact-checking assistant verifying a news STATEMENT.

Rules:
- ALWAYS use the search tool to find corroborating or contradicting sources
- Search at least twice with different queries
- NEVER rely only on prior knowledge
- Return label="yes" if FAKE/unverifiable, label="no" if REAL/confirmed
- Include 1-3 real source URLs (not Wikipedia)"""

SYSTEM_URL = """You are a strict fact-checking assistant verifying a news ARTICLE from a URL.

You have been given the article text. Your job:
1. Identify the main factual claims in the article
2. Use the search tool to verify each claim against OTHER news sources
3. For each claim, note whether it is confirmed or contradicted elsewhere
4. Return:
   - label="yes" if the article contains fake/misleading/unverified claims
   - label="no" if the article's claims are confirmed by other credible sources
   - reason: summary of what's real and what's not
   - evidence_links: URLs from your searches that confirm or contradict the article"""


def analyze(text: str, system_prompt: str) -> dict:
    model = build_llm()
    search = DuckDuckGoSearchRun()
    model_with_search = model.bind_tools([search])

    messages = [
        {"role": "system", "content": system_prompt},
        HumanMessage(content=text),
    ]

    response = model_with_search.invoke(messages)

    if not response.tool_calls:
        search_result = search.invoke(text[:300])
        messages.append(ToolMessage(content=search_result, tool_call_id="manual_0"))
    else:
        messages.append(response)
        for tc in response.tool_calls:
            result = search.invoke(tc["args"]["query"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Second search pass if only one was done
    if len([m for m in messages if isinstance(m, ToolMessage)]) < 2:
        second_query = f"fact check {text[:200]}"
        result2 = search.invoke(second_query)
        messages.append(ToolMessage(content=result2, tool_call_id="manual_1"))

    messages.append({
        "role": "user",
        "content": "Now give your final classification as structured output.",
    })

    structured = model.with_structured_output(FakeNewsResult)
    return structured.invoke(messages)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Fake News Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/check", response_model=FakeNewsResult)
def check(req: CheckRequest):
    raw = req.input.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Input is empty")

    if is_url(raw):
        article_text = fetch_article_text(raw)
        prompt = (
            f"Article URL: {raw}\n\n"
            f"Article content:\n{article_text}\n\n"
            "Verify the claims in this article against other news sources."
        )
        result = analyze(prompt, SYSTEM_URL)
        result.input_type = "url"
    else:
        prompt = f"Verify this news statement:\n\n{raw}"
        result = analyze(prompt, SYSTEM_TEXT)
        result.input_type = "text"

    return result


@app.get("/health")
def health():
    return {"status": "ok"}
