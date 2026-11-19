import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from mesquareassistantbot.chatbot import (
    Message,
    chat_with_bot,
    new_history,
    token_report,
)


def show_token_report(message: Message, history: list[Message]) -> None:
    report = token_report(message, history)
    if report:
        for line in report.splitlines():
            st.caption(line)


st.set_page_config(page_title="MeSquare Assistant")
st.title("MeSquare Assistant")
st.caption(
    "Ask about biomarkers, lab results, cardiovascular health and heart-healthy "
    "lifestyle. General information only, not medical advice. I am not a doctor."
)

if "history" not in st.session_state:
    st.session_state.history = new_history()

for message in st.session_state.history:
    if isinstance(message, SystemMessage):
        continue
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.text)
        if role == "assistant":
            show_token_report(message, st.session_state.history)

prompt = st.chat_input("Message the assistant")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = chat_with_bot(prompt, st.session_state.history)
        st.markdown(reply)
        show_token_report(st.session_state.history[-1], st.session_state.history)
