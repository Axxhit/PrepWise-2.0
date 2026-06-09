import os
import json
import requests
from google.genai import Client
from tavily import TavilyClient
from dotenv import load_dotenv
from google.genai import types


load_dotenv()

gemini = Client(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

FASTAPI_BASE = "http://localhost:8000"  # for calling our own backend tools in Step 2

# ── tool definitions 
#tool 1 : web search-----------------

def search_web(query: str) -> str:
    results = tavily.search(query = query, max_results = 3)
    simplified = [
        {"title": r["title"], "summary": r["content"][:200]}
        for r in results["results"]
    ]
    return json.dumps(simplified)


#---tool 2: evaluate answer-------------
def evaluate_answer(question: str, answer: str) -> str:
    prompt = f"""Score this interview answer 1-5 and give one line of feedback.

Question: {question}
Answer: {answer}

Respond ONLY as JSON: {{"score": <int>, "feedback": <string>}}"""
    
    response = gemini.models.generate_content(
        model="gemini-3.5-flash",
        contents = prompt
    )
    raw = response.text.strip().replace("'''json","").replace("'''","")
    return raw

#-----tool 3: fetch resume context-----------------
def fetch_resume_context(topic: str, session_id: str) -> str:
    try:
        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"q": topic, "session_id": session_id},
        )
        data = res.json()
        print(f"DEBUG retrieve response: {data}")  # add this line

        if "results" not in data:
            return json.dumps({"error": f"Unexpected response: {data}"})
        
        chunks = [r["chunk"] for r in data["results"]]
        return json.dumps({"context": chunks})
    except Exception as e:
        return json.dumps({"error": str(e)})
    

# tool registry — maps name → function
def make_tool_registry(session_id: str) -> dict:
    return {
        "search_web": search_web,
        "evaluate_answer": evaluate_answer,
        "fetch_resume_context": lambda topic: fetch_resume_context(topic, session_id),
    }


#---tool schemas------

# ── tool schemas for gemini ───────────────────────────────────
tool_schemas = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_web",
            description="Search the web for current information about a topic. Use sparingly, only when you need up-to-date info.",
            parameters=types.Schema(
                type= "object",
                properties={
                    "query": types.Schema(type="STRING", description="search query"),
                },
                required=["query"]
            )
        ),

        types.FunctionDeclaration(
            name = "evaluate_answer",
            description = "Evaluates a candidate's interview answer. returns score 1-5 and feedback.",
            parameters = types.Schema(
                type = 'OBJECT',
                properties = {
                    "question" : types.Schema(type="STRING", description="the interview question "),
                    "answer" : types.Schema(type="STRING", description="the candidate's answer")
                },
                required = ["question", "answer"]
            )
        ),

        types.FunctionDeclaration(
            name = "fetch_resume_context",
            description = "Fetches relevant resume sections for a given topic using RAG.",
            parameters = types.Schema(
                type = "OBJECT",
                properties = {
                    "topic": types.Schema(type="STRING", description="topic to search in resume")
                },
                required = ["topic"]
            )
        )
    ])
]


# ── updated agent loop ────────────────────────────────────────────
def run_agent(user_message: str,session_id: str,  max_iterations: int = 5):
    print(f"\n{'='*50}")
    print(f"USER: {user_message}")
    print(f"{'='*50}")

    TOOLS = make_tool_registry(session_id)
    history = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        response = gemini.models.generate_content(
            model="gemini-3.5-flash",
            contents=history,
            config=types.GenerateContentConfig(tools=tool_schemas)
        )

        candidate = response.candidates[0].content
        history.append(candidate)

        tool_calls = [p for p in candidate.parts if p.function_call]

        if not tool_calls:
            final = " ".join(p.text for p in candidate.parts if p.text)
            print(f"\nAGENT FINAL RESPONSE:\n{final}")
            return final

        tool_results = []
        for part in tool_calls:
            fn = part.function_call
            name = fn.name
            args = dict(fn.args)
            print(f"  → calling tool: {name}({args})")
            result = TOOLS[name](**args) if name in TOOLS else json.dumps({"error": "unknown tool"})
            print(f"  ← result: {result[:100]}...")
            tool_results.append(types.Part(
                function_response=types.FunctionResponse(name=name, response={"result": result})
            ))

        history.append(types.Content(role="user", parts=tool_results))

    return "Max iterations reached."


# ── test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    SESSION_ID = input("Enter your session_id from /embed: ").strip()
    run_agent(
        user_message=(
            "Fetch my resume context for 'machine learning', "
            "search the web for 'common ML engineer interview questions 2026', "
            "then evaluate this answer to the question 'Explain your ML project': "
            "'I built BARI, an Android app using Random Forest achieving 96% accuracy "
            "with zero false negatives on at-risk class detection'"
        ),
        session_id=SESSION_ID
    )