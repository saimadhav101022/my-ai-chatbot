import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.rerun()

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

st.title("CAM_VISION")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask me anything...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    response = client.models.generate_content(
      model="gemini-3.5-flash-lite",
        contents=question
    )

    answer = response.text

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)