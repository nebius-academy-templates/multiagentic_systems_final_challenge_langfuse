# Final Practice — Open Challenge: Design Your Own Multi-Agent LangGraph Pipeline

In this task, you'll design and build a LangGraph pipeline **of your choice**. Unlike the guided QA agent exercise, this challenge starts with a blank canvas — you pick the problem domain, decompose the work into agents, and wire everything together yourself.

## **Minimum requirements for your pipeline**

| Constraint | Requirement |
| --- | --- |
| Agent nodes | At least two specialist agent nodes, each with a clearly bounded responsibility |
| Human-in-the-loop | Exactly one `interrupt()`-based review node that can loop back |
| MCP server | At least one MCP server providing external context (Notion, GitHub, Slack, filesystem, etc.) |
| State | A single `TypedDict` state shared across all nodes |
| Checkpointer | `MemorySaver` so the interrupt/resume loop persists across calls |

---

## 1. Choose your challenge

Pick one of the example domains below or propose your own.

### 📋 Example domains

| Domain | What it does |
| --- | --- |
| **Code review assistant** | Reads a PR diff, queries an internal style-guide MCP, generates review comments, receives human approval/edits, posts comments to GitHub |
| **Content pipeline** | Fetches a GitHub issue, searches Notion for related docs, drafts a blog post, receives human approval, publishes as a Notion page |
| **Incident responder** | Pulls an on-call alert, searches a runbook MCP server, drafts a post-mortem, receives human confirmation of severity, posts to Slack |
| **Onboarding generator** | Reads a new-hire Jira ticket, fetches team docs via Notion MCP, drafts a personalized onboarding plan, receives human feedback, saves to Notion |
| **Meeting summarizer** | Reads a transcript file, queries a Confluence MCP for project context, writes action items, receives human edits, sends to email or Slack |
| **Your own idea** | Anything that has a multi-step information-gathering phase, a generation phase, and a human gate |

> **Think before you continue.**
> 
> 
> Write down (on paper or in a comment block) a one-paragraph description of:
> 
> - What problem this pipeline solves
> - Who the end user is
> - What "done" looks like — what artifact is produced and where it lands

---

## 2. Define your agent role decomposition

Before writing any code, answer these questions. There is no single right answer — the goal is intentional design.

### 2.1 Identify responsibilities

List every distinct responsibility your pipeline needs to carry out. Think in verbs: *fetch*, *search*, *draft*, *review*, *validate*, *publish*.

> **Checkpoint — ask yourself:**
> 
> - Which responsibilities require external data (API calls, MCP lookups)?
> - Which responsibilities require LLM reasoning?
> - Which responsibilities require a human decision?

### 2.2 Group into agents

Group the responsibilities from 2.1 into **at least two agents**. Each agent should have a single, coherent job that could be described in one sentence.

Use this template for each agent:

```
Agent name  : _______________
One-sentence responsibility : _______________
Inputs from state : _______________
Outputs written to state : _______________
LLM calls it makes : ___ (count and brief description)
External tool calls it makes : ___
```

> **Common mistake to avoid:** do not put data-fetching and generation in the same agent. Keep gathering and reasoning separate so each agent stays testable and replaceable.
> 

### 2.3 Decide the human gate

Answer:

- After which agent does the human review happen, and why at that point?
- What does the human see (which state field is printed)?
- What are the two outcomes — what do "approve" and "request changes" mean?
- Where does the graph route for each outcome?

### 2.4 Sketch the graph (on paper or ASCII)

Draw the nodes, edges, and the conditional branch before touching code. Example structure:

```
START → [Agent A] → [Agent B] → [Human Review]
                                      │
                    ┌─────────────────┴─────────────────┐
                    │ approved                           │ needs changes
                    ▼                                    ▼
               [Publish] → END                     [Agent B]  (loop)
```

Your graph does not have to match this shape — it just needs to be intentional.

---

## 3. Set up your environment

Once you have your design:

1. Create a new folder for your project in your copy of [`multiagentic_systems_final_challenge_langfuse`](https://github.com/nebius-academy-templates/multiagentic_systems_final_challenge_langfuse).
2. Copy `requirements.txt` as a starting point and add any extra packages your domain needs.
3. Create a `.env` file with the API keys your pipeline will use, including `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (from your Langfuse project's **Settings** → **API Keys**) and `LANGFUSE_BASE_URL=https://cloud.langfuse.com` for tracing, and `ANTHROPIC_OPEN_TELEMETRY=false` so each Claude call shows once in the trace.

Minimum `requirements.txt` (Python 3.11 or newer):

```
langchain>=0.3
langchain-anthropic>=1.7
langchain-core>=1
langchain-mcp-adapters>=0.1
langgraph>=0.2
python-dotenv>=1.0
httpx>=0.27
langfuse>=4.7
```

1. Confirm Node.js is available for MCP servers that use `npx`:

```bash
node --version   # should print v24 or higher
```

---

## 4. Create `state.py`

Define your `TypedDict`. Use your decomposition notes from step 2 to decide which fields belong here.

### Guidelines

- Every piece of data passed between nodes lives in state — nothing is passed as function arguments.
- Give each field a name that makes its owner obvious (e.g. `context_agent` writes `raw_docs`; `draft_agent` writes `draft_output`).
- Add `retry_count: int` to support the loop safety exit.
- Add `human_approved: bool` and `human_feedback: str` for the HITL node.

```python
from typing import TypedDict

class YourState(TypedDict):
    # inputs
    ...
    # per-agent outputs
    ...
    # HITL fields
    human_approved: bool
    human_feedback: str
    # control
    retry_count: int
    published: bool
```

> **Checkpoint:** Does every field have exactly one node that writes it? If two nodes write the same field, consider splitting them or using `Annotated[list, operator.add]` for accumulation.
> 

---

## 5. Configure your MCP server

### 5.1 Choose your MCP server

Decide which MCP server gives your pipeline the external context it needs.

| Server | npm package | Use case |
| --- | --- | --- |
| Notion | `@notionhq/notion-mcp-server` | Docs, wikis, project pages |
| Filesystem | `@modelcontextprotocol/server-filesystem` | Local files, runbooks |

### 5.2 Create `mcp_config.json`

```json
{
  "your_server_name": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "<npm-package-name>"],
    "env": {
      "YOUR_API_KEY_ENV_VAR": "your_key_here"
    }
  }
}
```

> **Think:** which tools does your MCP server expose? Which tool names will your agents call to search and which to fetch full content? (Use `client.get_tools()` in a scratch script to print them out.)
> 

---

## 6. Implement agent nodes in `nodes.py`

### Shared setup

```python
import asyncio

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from state import YourState

_llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=16000, streaming=True)

def _call(system: str, user: str) -> str:
    return _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).text
```

### For each agent node

Follow this pattern:

```python
def your_agent_node(state: YourState, config: RunnableConfig) -> dict:
    """One-sentence description of this agent's job."""
    # 1. Read the fields this agent needs from `state`
    # 2. Make LLM / tool calls
    # 3. Return only the fields this agent writes
    return {"field_a": ..., "field_b": ...}
```

> **Important constraints:**
> 
> - An agent node must return **only** the state fields it owns (see your decomposition notes).
> - Never read from a field your agent does not own without a clear reason.
> - If your agent calls MCP tools, receive `notion_tools` (or equivalent) from `config["configurable"]`, and type the parameter as `config: RunnableConfig` so LangGraph passes it in.
> - MCP tools run only asynchronously. From a sync node, call them with `asyncio.run(tool.ainvoke({...}))`.

### MCP tool call pattern

```python
notion_tools = config["configurable"]["notion_tools"]

# find the right tool by name
search_tool = next(t for t in notion_tools if "search" in t.name)
results = asyncio.run(search_tool.ainvoke({"query": "your query"}))

# parse page IDs from results, then fetch each page
fetch_tool = next(t for t in notion_tools if t.name == "API-retrieve-a-page")
page = asyncio.run(fetch_tool.ainvoke({"page_id": page_id}))
```

### Human review node

```python
from langgraph.types import interrupt

def human_review_node(state: YourState) -> dict:
    # 1. Print the content the human needs to review
    print(state["the_field_to_review"])
    # 2. Pause and collect feedback
    response = interrupt("Your prompt here — approve or describe changes: ")
    # 3. Route based on response
    approved = response.strip().lower() in {"approve", "ok", "yes", ""}
    return {
        "human_approved": approved,
        "human_feedback": "" if approved else response.strip(),
    }
```

### Publish node

```python
def publish_node(state: YourState) -> dict:
    # Post, save, or send the final artifact
    ...
    return {"published": True}
```

---

## 7. Wire the graph in `graph.py`

```python
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from nodes import agent_a_node, agent_b_node, human_review_node, publish_node
from state import YourState

def review_router(state: YourState) -> str:
    if state.get("human_approved") or state.get("retry_count", 0) >= 3:
        return "publish"
    return "agent_b"   # ← replace with the node that should re-run

def build_graph():
    checkpointer = MemorySaver()
    graph = StateGraph(YourState)

    # Register nodes
    graph.add_node("agent_a", agent_a_node)
    graph.add_node("agent_b", agent_b_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("publish", publish_node)

    # Edges
    graph.set_entry_point("agent_a")
    graph.add_edge("agent_a", "agent_b")
    graph.add_edge("agent_b", "human_review")
    graph.add_conditional_edges("human_review", review_router, {"publish": "publish", "agent_b": "agent_b"})
    graph.add_edge("publish", END)

    return graph.compile(checkpointer=checkpointer)
```

> **Checklist before moving on:**
> 
> - [ ]  Every node registered with `add_node`?
> - [ ]  Entry point set?
> - [ ]  No node is an island (every node has at least one outgoing edge)?
> - [ ]  Conditional edge covers all possible return values of the router?
> - [ ]  Safety exit via `retry_count >= 3`?

---

## 8. Implement `main.py`

```python
import sys, uuid, json, asyncio
from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from graph import build_graph


def load_mcp_tools() -> list:
    with open("mcp_config.json") as f:
        cfg = json.load(f)
    client = MultiServerMCPClient(cfg)
    return asyncio.run(client.get_tools())


def stream_until_pause(graph, payload, thread) -> str | None:
    """Stream the graph until it ends or pauses. Return the interrupt prompt, or None at the end."""
    for step in graph.stream(payload, config=thread, stream_mode="updates"):
        node_name, updates = next(iter(step.items()))
        if node_name == "__interrupt__":
            return updates[0].value
        print(f"  ✓ {node_name} wrote: {list(updates.keys())}")
    return None


def run(your_input: str) -> None:
    mcp_tools = load_mcp_tools()
    graph = build_graph()
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "notion_tools": mcp_tools,   # rename key to match your agent node
        },
        "callbacks": [CallbackHandler()],
    }

    initial = {
        # populate all fields from your TypedDict
        "human_approved": False,
        "human_feedback": "",
        "published": False,
        "retry_count": 0,
    }

    print(f"\n{'='*60}\nRunning pipeline for: {your_input}\n{'='*60}")

    prompt = stream_until_pause(graph, initial, thread)
    while prompt is not None:
        response = input(f"\n[REVIEW] {prompt}\n> ").strip()
        prompt = stream_until_pause(graph, Command(resume=response), thread)

    final = graph.get_state(thread).values
    print(f"\n  published = {final.get('published')}")
    get_client().flush()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "default-input")
```

---

## 9. Run and validate

```bash
python main.py <your-input>
```

Work through this checklist as you test:

| Check | What to look for |
| --- | --- |
| MCP tools load | No import error; `load_mcp_tools()` returns a non-empty list |
| Agent A runs | Prints `✓ agent_a wrote: [...]` with the expected fields |
| Agent B runs | Prints `✓ agent_b wrote: [...]` with the expected fields |
| Interrupt fires | Prompt appears in the terminal; execution pauses |
| Feedback loop | Entering feedback re-runs the generation agent; `retry_count` increments |
| Approval | Entering "approve" routes to the publish node |
| Publish | `published = True` printed at the end |
| Langfuse trace | Open [cloud.langfuse.com](https://cloud.langfuse.com/), go to **Tracing** and confirm the full run is visible, one trace per `stream()` call |

> 💡 Filter traces by name to `LangGraph`, so the list shows one row per `stream()` call. A trace that ends at the review pause shows as an error: that's the pause, not a failure.

---

## Submission checklist


- [ ]  `mcp_config.json` configured with your chosen MCP server
- [ ]  `state.py` — `TypedDict` with all fields, including `human_approved`, `human_feedback`, `retry_count`, `published`
- [ ]  At least two agent nodes, each with a single clearly bounded responsibility
- [ ]  Agent nodes only write the state fields they own
- [ ]  One human review node that calls `interrupt()` and returns both `human_approved` and `human_feedback`
- [ ]  One publish node that produces the final artifact
- [ ]  `graph.py` — all nodes registered, edges complete, conditional edge covers all router return values
- [ ]  `retry_count >= 3` safety exit present
- [ ]  Compiled with `MemorySaver`
- [ ]  `main.py` — MCP tools passed via `config["configurable"]`; interrupt/resume loop works
- [ ]  Full end-to-end run completes without errors
- [ ]  Human feedback loop works: pause → feedback → agent re-runs → approve
- [ ]  Final artifact published after approval
- [ ]  Code committed and pushed to `main`
