import sys
sys.path.append('/Users/abeburton/Desktop/census_llm/src/')
import server.query_api as q
from openai import OpenAI
from openai import APIConnectionError
from dotenv import load_dotenv
import streamlit as st
import os

# main paige content allowing for queries and data display
st.title("Census Query Bot")
input = st.text_input("Ask the Census Bot what you want to know from Census Data!")
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
    st.markdown('**Enter your OpenAI API Key**')
    user_openai_key = st.text_input("For accessing GPT")
    st.write('*Get your OpenAI key here* :thumbsup: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key')
    st.markdown('**Enter your Census API Key**')
    user_census_key = st.text_input("For accessing Census Data")
    st.write('*Get your Census API key here* :thumbsup: https://api.census.gov/data/key_signup.html')

    if user_openai_key == '':
        st.error(f'Please enter your OpenAI API Key!', icon="🚨")
    if user_census_key == '':
        st.error(f'Please enter your Census API Key!', icon="🚨")
        
    st.markdown("**Site Usage:**")
    st.markdown("""Put any question you want answered using Census/ACS
                 Data and this site will query the Census and give you data to
                 answer your question.""")
    st.markdown("""Examples: How many men are under 45 in
                 Connectict? How many people commute to work via public transit
                 in New York versus Los Angeles?""")
