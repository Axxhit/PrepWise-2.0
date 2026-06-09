import os
import json
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from context_manager import trim_messages, log_token_usage
from llm_client import get_client, BaseLLMClient

SYSTEM_PROMPT = """You are a technical interviewer for ML engineering roles.
Use the available tools to fetch resume context, search for relevant questions,
and evaluate candidate answers. Be specific and reference actual project details."""


load_dotenv()

llm = get_client(os.getenv("LLM_PROVIDER", "gemini"))  # reads from env var LLM_PROVIDER
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
FASTAPI_BASE = "http://localhost:8000"

# ── state schema ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    session_id: str
    iteration: int

# ── same 3 tools from step 2 ──────────────────────────────────
def search_web(query: str) -> str:
    results = tavily.search(query=query, max_results=3)
    simplified = [
        {"title": r["title"], "summary": r["content"][:200]}
        for r in results["results"]
    ]
    return json.dumps(simplified)


def evaluate_answer(question: str, answer: str) -> str:
    prompt = f"""Score this interview answer 1-5 and give one line of feedback.

Question: {question}
Answer: {answer}

Respond ONLY as JSON: {{"score": <int>, "feedback": "<string>"}}"""
    response = gemini.models.generate_content(
        model="gemini-3.5-flash", contents=prompt
    )
    return response.text.strip().replace("```json", "").replace("```", "")


def fetch_resume_context(topic: str, session_id: str) -> str:
    try:
        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"session_id": session_id, "q": topic}
        )
        data = res.json()
        if "results" not in data:
            return json.dumps({"error": f"Unexpected response: {data}"})
        chunks = [r["chunk"] for r in data["results"]]
        return json.dumps({"context": chunks})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_REGISTRY = {
    "search_web": search_web,
    "evaluate_answer": evaluate_answer,
    "fetch_resume_context": fetch_resume_context,
}

tool_schemas = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_web",
            description="Searches the web for current information.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING")},
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="evaluate_answer",
            description="Evaluates a candidate's interview answer. Returns score 1-5.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(type="STRING"),
                    "answer": types.Schema(type="STRING")
                },
                required=["question", "answer"]
            )
        ),
        types.FunctionDeclaration(
            name="fetch_resume_context",
            description="Fetches relevant resume sections for a topic using RAG.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "topic": types.Schema(type="STRING"),
                    "session_id": types.Schema(type="STRING")
                },
                required=["topic", "session_id"]
            )
        )
    ])
]

# ── node 1: call LLM ──────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    print(f"\n--- LLM node (iteration {state['iteration']}) ---")

    trimmed = trim_messages(
        state["messages"],
        max_tokens=6000,
        system_prompt=SYSTEM_PROMPT
    )
    log_token_usage(trimmed, label=f"iteration {state['iteration']}")

    # build history for gemini format (tool-calling still needs raw genai)
    history = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        if isinstance(m["content"], str)
        else types.Content(role=m["role"], parts=m["content"])
        for m in trimmed
    ]

    # tool-calling requires genai directly — use abstracted client for
    # non-tool calls only (question gen, evaluation, summarization)
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        # full tool-calling support
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(
                tools=tool_schemas,
                system_instruction=SYSTEM_PROMPT,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
            )
        )
        candidate = response.candidates[0].content
        tool_calls = [p for p in candidate.parts if p.function_call]

        if tool_calls:
            return {
                "messages": [{"role": "model", "content": candidate.parts}],
                "session_id": state["session_id"],
                "iteration": state["iteration"] + 1
            }
        final_text = " ".join(p.text for p in candidate.parts if p.text)

    else:
        # llama/mixtral via groq — no tool calling, single prompt
        # flatten history to one prompt string
        flat_prompt = SYSTEM_PROMPT + "\n\n"
        for m in trimmed:
            role = m["role"].upper()
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            flat_prompt += f"{role}: {content}\n"
        flat_prompt += "ASSISTANT:"

        result = llm.chat(flat_prompt, temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")))
        final_text = result["text"]
        print(f"  [{result['model']}] {result['input_tokens']} in / {result['output_tokens']} out tokens")

    return {
        "messages": [{"role": "model", "content": final_text}],
        "session_id": state["session_id"],
        "iteration": state["iteration"] + 1
    }


# ── node 2: execute tools ─────────────────────────────────────
def call_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    parts = last_message["content"]

    tool_results = []
    for part in parts:
        if not part.function_call:
            continue
        fn = part.function_call
        name = fn.name
        args = dict(fn.args)
        print(f"  → tool: {name}({args})")
        result = TOOL_REGISTRY[name](**args) if name in TOOL_REGISTRY else json.dumps({"error": "unknown tool"})
        print(f"  ← result: {result[:80]}...")
        tool_results.append(
            types.Part(function_response=types.FunctionResponse(
                name=name, response={"result": result}
            ))
        )

    return {
        "messages": [{"role": "user", "content": tool_results}],
        "session_id": state["session_id"],
        "iteration": state["iteration"]
    }


# ── conditional edge: route or end ───────────────────────────
def should_continue(state: AgentState) -> str:
    if state["iteration"] >= 5:
        print("Max iterations reached.")
        return END

    last = state["messages"][-1]
    content = last["content"]

    # if content is a string → final answer → end
    if isinstance(content, str):
        return END

    # if content has tool calls → continue
    has_tool_calls = any(
        hasattr(p, "function_call") and p.function_call
        for p in content
    )
    return "call_tools" if has_tool_calls else END

# ── build graph ───────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm)
    graph.add_node("call_tools", call_tools)

    graph.set_entry_point("call_llm")

    graph.add_conditional_edges("call_llm", should_continue, {
        "call_tools": "call_tools",
        END: END
    })

    graph.add_edge("call_tools", "call_llm")

    return graph.compile()

# ── run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    SESSION_ID = input("Enter session_id: ").strip()

    app = build_graph()

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": (
            f"Fetch my resume context for 'machine learning' (session_id: {SESSION_ID}), "
            "then evaluate this answer to 'Explain your ML project': "
            "'I built BARI using Random Forest with 96% accuracy and zero false negatives'"
        )}],
        "session_id": SESSION_ID,
        "iteration": 0
    }

    final = app.invoke(initial_state)
    print("\n=== FINAL STATE ===")
    last = final["messages"][-1]["content"]
    print(last if isinstance(last, str) else "[tool parts]")