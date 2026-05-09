
import streamlit as st

from langchain_core.messages import HumanMessage
from agent import graph


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🤖",
    layout="centered"
)


# =========================
# TITLE
# =========================
st.title("🤖 LangGraph Multi-Agent AI Assistant")
st.markdown("Built with LangGraph + Ollama + ReAct")


# =========================
# SESSION STATE
# =========================
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "message_type": None,
        "next": None,
        "scratchpad": "",
        "react_steps": 0,
        "notes": [],
    }


# =========================
# DISPLAY CHAT HISTORY
# =========================
for msg in st.session_state.state["messages"]:

    if msg.type == "human":
        with st.chat_message("user"):
            st.write(msg.content)

    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.write(msg.content)


# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Type your message...")


# =========================
# CHAT FLOW
# =========================
if user_input:

    # show user message
    with st.chat_message("user"):
        st.write(user_input)

    # add to graph state
    st.session_state.state["messages"].append(
        HumanMessage(content=user_input)
    )

    # run graph
    st.session_state.state = graph.invoke(
        st.session_state.state
    )

    # get latest AI message
    ai_message = st.session_state.state["messages"][-1].content

    # show assistant response
    with st.chat_message("assistant"):
        st.write(ai_message)