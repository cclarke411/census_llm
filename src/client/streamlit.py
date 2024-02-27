import sys
sys.path.append('/Users/abeburton/Desktop/census_llm/src/')
import server.query_api as q
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import os

# main paige content allowing for queries and data display
st.title("Census Query Bot")
input = st.text_input("Ask the Census Bot what you want to know!")
@st.cache_data
def load_data():
    query = q.Query(q.api_access_url, q.variables, q.geography_fips)
    df = query.format_data()
    explanation = query.explanation()
    
    return df, explanation
info = load_data()
st.dataframe(info[0])
st.text(info[1])


# sidebar with helpful chatbot
load_dotenv()
with st.sidebar:
    st.title("GPT Clone To explain site")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
