# chat_512k.py – interface Web 512k
import streamlit as st
from v8_chat_512k import generer_512k

st.set_page_config(page_title="RâS-Fr Chat 512k", layout="wide")
st.title("💬 RâS-Fr Chat – 512 000 tokens – 4 bits – < 6 GB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Votre message…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # génération 512k
    context = "\n".join([m["content"] for m in st.session_state.messages])
    with st.spinner("Génération…"):
        ids = generer_512k(context, max_tokens=200, temperature=0.9)
        réponse = "".join([chr(i % 256) for i in ids])[:200]
    st.session_state.messages.append({"role": "assistant", "content": réponse})
    st.chat_message("assistant").write(réponse)