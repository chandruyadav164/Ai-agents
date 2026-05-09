from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from pydantic import BaseModel
from ddgs import DDGS

from datetime import datetime
import ast
import operator as op
import re
import random


# =========================
# LLM
# =========================
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)


# =========================
# CLASSIFIER SCHEMA
# =========================
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"]


# =========================
# STATE
# =========================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None
    scratchpad: str
    react_steps: int
    notes: list[str]


# =========================
# SAFE CALCULATOR
# =========================
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}


def safe_calculator(expr: str):
    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            operator = _ALLOWED_OPS.get(type(node.op))

            if not operator:
                raise ValueError("Unsupported operator")

            return operator(left, right)

        if isinstance(node, ast.UnaryOp):
            operator = _ALLOWED_OPS.get(type(node.op))

            if not operator:
                raise ValueError("Unsupported unary operator")

            return operator(eval_node(node.operand))

        raise ValueError("Invalid calculation")

    tree = ast.parse(expr, mode="eval")
    return eval_node(tree.body)


# =========================
# TOOLS
# =========================
def search_tool(query: str):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        return "No search results found."

    formatted = []

    for item in results:
        title = item.get("title", "")
        body = item.get("body", "")
        href = item.get("href", "")
        formatted.append(f"Title: {title}\nSummary: {body}\nURL: {href}")

    return "\n\n".join(formatted)


def calculator_tool(expression: str):
    try:
        return str(safe_calculator(expression))
    except Exception:
        return "Invalid calculation. Use plain math like 2 + 2 * 5."


def time_tool(_: str = ""):
    return datetime.now().strftime("%H:%M:%S")


def date_tool(_: str = ""):
    return datetime.now().strftime("%Y-%m-%d")


def datetime_tool(_: str = ""):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wikipedia_tool(query: str):
    search_query = f"{query} wikipedia"

    with DDGS() as ddgs:
        results = list(ddgs.text(search_query, max_results=2))

    if not results:
        return "No Wikipedia-style result found."

    item = results[0]
    return f"{item.get('title', '')}\n{item.get('body', '')}\n{item.get('href', '')}"


def joke_tool(_: str = ""):
    jokes = [
        "Why did the programmer quit his job? Because he did not get arrays.",
        "Why do Python developers wear glasses? Because they cannot C.",
        "I told my computer I needed a break, and it said: no problem, I will go to sleep.",
    ]

    return random.choice(jokes)


def notes_tool(action_input: str, notes: list[str]):
    text = action_input.strip()

    if text.lower().startswith("add "):
        note = text[4:].strip()
        notes.append(note)
        return f"Note saved: {note}", notes

    if text.lower() in ["show", "list", "get"]:
        if not notes:
            return "No notes saved yet.", notes

        return "\n".join(f"{i + 1}. {note}" for i, note in enumerate(notes)), notes

    return "Use notes like: add buy milk OR show", notes


def run_tool(action: str, action_input: str, state: State):
    notes = state.get("notes", [])

    if action == "search":
        return search_tool(action_input), notes

    if action == "calculator":
        return calculator_tool(action_input), notes

    if action == "time":
        return time_tool(action_input), notes

    if action == "date":
        return date_tool(action_input), notes

    if action == "datetime":
        return datetime_tool(action_input), notes

    if action == "wikipedia":
        return wikipedia_tool(action_input), notes

    if action == "joke":
        return joke_tool(action_input), notes

    if action == "notes":
        return notes_tool(action_input, notes)

    return "Invalid action.", notes


# =========================
# CLASSIFIER
# =========================
def classify_message(state: State):
    last_message = state["messages"][-1]

    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """
Classify the user message.

Return emotional if the user talks about:
sadness, stress, anxiety, fear, relationship pain, loneliness, motivation, feelings.

Return logical if the user asks about:
facts, search, time, date, math, coding, memory, notes, explanation, chatbot tasks, general questions.
"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])

    return {"message_type": result.message_type}


# =========================
# ROUTER
# =========================
def router(state: State):
    if state["message_type"] == "emotional":
        return {"next": "therapist"}

    return {
        "next": "react",
        "scratchpad": "",
        "react_steps": 0,
    }


# =========================
# THERAPIST AGENT
# =========================
def therapist_agent(state: State):
    last_message = state["messages"][-1]

    reply = llm.invoke([
        {
            "role": "system",
            "content": """
You are a supportive therapist-style chatbot.
Be warm, calm, and helpful.
Do not diagnose.
If the user mentions self-harm or danger, encourage immediate help from emergency services or trusted people.
"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])

    return {
        "messages": [AIMessage(content=reply.content)]
    }


# =========================
# PARSER
# =========================
def parse_react_response(text: str):
    if "Final:" in text:
        return "final", text.split("Final:", 1)[1].strip(), ""

    action_match = re.search(
        r"Action:\s*(search|calculator|time|date|datetime|wikipedia|joke|notes)\s*",
        text,
        re.IGNORECASE,
    )

    input_match = re.search(
        r"Action Input:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not action_match:
        return "bad", "", ""

    action = action_match.group(1).lower()
    action_input = input_match.group(1).strip() if input_match else ""

    return "action", action, action_input


# =========================
# REACT AGENT
# =========================
def react_agent(state: State):
    query = state["messages"][-1].content
    scratchpad = state.get("scratchpad", "")
    react_steps = state.get("react_steps", 0)

    # If we already have a tool observation, force final answer.
    if scratchpad.strip():
        final_prompt = f"""
You are answering the user's question using the tool observation below.

Rules:
- Do not call any tool.
- Do not say "no new information" if useful information is present.
- Give a clear, helpful final answer.
- Summarize the observation in natural language.
- Include important URLs if relevant.

User Query:
{query}

Tool Work:
{scratchpad}

Final answer:
"""

        reply = llm.invoke(final_prompt).content.strip()

        return {
            "messages": [AIMessage(content=reply)],
            "scratchpad": "",
            "react_steps": 0,
        }

    if react_steps >= 5:
        return {
            "messages": [
                AIMessage(content="I could not complete the tool reasoning properly. Please try asking again more directly.")
            ],
            "scratchpad": "",
            "react_steps": 0,
        }

    prompt = f"""
You are a STRICT ReAct chatbot agent.

Available tools:

1. search
Use for latest/current facts, news, web information.

2. calculator
Use for math.

3. time
Use for current time.

4. date
Use for current date.

5. datetime
Use for current date and time.

6. wikipedia
Use for encyclopedia-style information.

7. joke
Use when user asks for a joke.

8. notes
Use for memory notes.

Rules:
- Use exactly one tool if needed.
- If no tool is needed, return Final.
- Do not repeat the same tool call.
- Action must be exactly one of:
  search
  calculator
  time
  date
  datetime
  wikipedia
  joke
  notes

Format exactly:

Thought: short reasoning
Action: tool_name
Action Input: input text

OR:

Final: answer

User Query:
{query}
"""

    response = llm.invoke(prompt).content.strip()

    print("\n[ReAct]")
    print(response)

    kind, value1, value2 = parse_react_response(response)

    if kind == "final":
        return {
            "messages": [AIMessage(content=value1)],
            "scratchpad": "",
            "react_steps": 0,
        }

    if kind == "bad":
        return {
            "messages": [AIMessage(content=response)],
            "scratchpad": "",
            "react_steps": 0,
        }

    action = value1
    action_input = value2

    observation, updated_notes = run_tool(action, action_input, state)

    print("\n[Observation]")
    print(observation)

    new_scratchpad = (
        response
        + f"\nObservation: {observation}\n"
    )

    return {
        "scratchpad": new_scratchpad,
        "react_steps": react_steps + 1,
        "notes": updated_notes,
    }



# =========================
# LOOP CONTROL
# =========================
def should_continue(state: State):
    if state.get("messages") and state["messages"][-1].type == "ai":
        return END

    return "react"


# =========================
# GRAPH BUILD
# =========================
builder = StateGraph(State)

builder.add_node("classifier", classify_message)
builder.add_node("router", router)
builder.add_node("therapist", therapist_agent)
builder.add_node("react", react_agent)

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "router")

builder.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "therapist": "therapist",
        "react": "react",
    },
)

builder.add_conditional_edges(
    "react",
    should_continue,
    {
        "react": "react",
        END: END,
    },
)

builder.add_edge("therapist", END)

graph = builder.compile()


# =========================
# RUN CHATBOT
# =========================
def run_chatbot():
    state = {
        "messages": [],
        "message_type": None,
        "next": None,
        "scratchpad": "",
        "react_steps": 0,
        "notes": [],
    }

    print("Chatbot started. Type exit or quit to stop.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Bye")
            break

        state["messages"].append(HumanMessage(content=user_input))

        state = graph.invoke(state)

        if state.get("messages"):
            print("\nAssistant:", state["messages"][-1].content)



