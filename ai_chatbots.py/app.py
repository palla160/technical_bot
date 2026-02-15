import streamlit as st
from support_brain import support_graph   # <-- import your LangGraph file

st.set_page_config(page_title="AI Support Brain", page_icon="🤖")

st.title("🤖 AI Customer Support Brain")

# ---------------- SESSION MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- DISPLAY HISTORY ----------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask your support question...")

if user_input:
    # show user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # call LangGraph
    result = support_graph.invoke({
        "user_message": user_input
    })

    ai_reply = result.get("response", "No response")

    # show AI message
    st.session_state.chat_history.append(("assistant", ai_reply))
    with st.chat_message("assistant"):
        st.markdown(ai_reply)