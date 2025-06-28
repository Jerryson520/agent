# from agent import build_graph
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
import requests


st.title("ChatBot Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
prompt = st.chat_input("Hello, my name is Jerry, input any question you want to ask")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = requests.post(
            "http://localhost:8000/chat",
            json = {"question": prompt},
        )
        try:
            answer = response.json()['response']
            st.write(answer)
        except Exception as e:
            st.error(f"解析 response.json() 时出错：{e}")
            st.write(response.text)
            
    
    st.session_state.messages.append({"role": "assistant", "content": response})