# 🧠 Fake News Detection Agent (LangChain)

## 🎯 Objective

Build a Fake News Detection system using LangChain that classifies input news as:

* `"yes"` → Fake News
* `"no"` → Real News

The system must also provide **supporting evidence links** used to justify the classification.

---

## 🧱 Output Schema (STRICT)

Use structured output (Pydantic model):

```python
from pydantic import BaseModel, Field
from typing import List

class FakeNewsResult(BaseModel):
    label: str = Field(description="Return ONLY 'yes' if fake news, 'no' if real news")
    reason: str = Field(description="Short explanation of why the news is fake or real")
    evidence_links: List[str] = Field(description="List of URLs used to verify the claim")
```

````

---

## ⚠️ Rules (VERY IMPORTANT)

1. `label` must be strictly:
   - `"yes"` → Fake
   - `"no"` → Real  
   ❌ Do NOT return anything else (no "maybe", no "uncertain")

2. Always include **at least 1–3 evidence links**:
   - Prefer trusted sources (news, Wikipedia, official sites)
   - Avoid random blogs unless necessary

3. If no strong evidence is found:
   - Return `"yes"` (assume suspicious)
   - Explain uncertainty clearly

4. Keep `reason` concise (2–4 lines max)

---

## 🔍 System Design

### Step 1: Input
User provides a news statement/article.

---

### Step 2: Claim Extraction
Break the news into key factual claims.

---

### Step 3: Retrieval (MANDATORY)
Use a search tool (Tavily preferred) to:
- Fetch relevant articles
- Validate claims

---

### Step 4: Reasoning
LLM must:
- Compare claims with retrieved data
- Identify inconsistencies or confirmations

---

### Step 5: Final Classification
Return structured output using the schema.

---

## 🔧 Tools to Use

- Tavily Search (preferred)
- OR DuckDuckGo (fallback)

LangChain tools must be integrated.

---

## 🧪 Example

### Input:
"Government announces free electricity for all citizens starting tomorrow."

### Output:
```json
{
  "label": "yes",
  "reason": "No credible sources confirm this announcement. Government policy changes of this scale are always widely reported.",
  "evidence_links": [
    "https://example-news.com/article1",
    "https://example-news.com/article2"
  ]
}
````

---

## 🏗️ Implementation Requirements

* Use LangChain
* Use structured output (Pydantic)
* Use an LLM (OpenAI / Claude)
* Use a retrieval tool (Tavily)

---

## 🚀 Bonus (if possible)

* Add retry if no links found
* Add confidence score internally (optional, not in schema)
* Use LangGraph for multi-step flow (optional)

---

## ❌ What to Avoid

* Do NOT hallucinate links
* Do NOT skip retrieval step
* Do NOT return unstructured text
* Do NOT give vague answers

---

## ✅ Expected Behavior

The system should behave like a **fact-checking agent**, not just a classifier.

It must:

* Think → Search → Verify → Decide

---

## 📌 Final Note

Accuracy is more important than speed.
Always prioritize **trusted sources + reasoning** over guessing.
