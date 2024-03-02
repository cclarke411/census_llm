import sys
sys.path.append('/Users/abeburton/Desktop/census_llm/src/')
import server.query_api as q
import server.analysis as a
from openai import OpenAI
from openai import APIConnectionError
from dotenv import load_dotenv
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()
# check for API Keys:
environment_keys = False
if 'OPENAI_API_KEY' in os.environ and 'CENSUS_API_KEY' in os.environ:
    environment_keys = True


# sidebar with FAQ
load_dotenv()
with st.sidebar:
    st.markdown('**Enter your OpenAI API Key**')
    user_openai_key = st.text_input("For accessing GPT")
    st.write('*Get your OpenAI key here* :point_down: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key')
    st.markdown('**Enter your Census API Key**')
    user_census_key = st.text_input("For accessing Census Data")
    st.write('*Get your Census API key here* :point_down: https://api.census.gov/data/key_signup.html')

    if user_openai_key == '' and not environment_keys:
        st.error(f'Please enter your OpenAI API Key!', icon="🚨")
    if user_census_key == '' and not environment_keys:
        st.error(f'Please enter your Census API Key!', icon="🚨")
        
    expander1 = st.expander("**Data Source Information**")
    expander1.write("""
        This site has access to the ACS 1, ACS 3, ACS 5, and 2020 Decennial
        Census Datasets.
        
        The Decennial Census and the American Community Survey (ACS), both conducted by the U.S. Census Bureau, 
        collect different types of data about the population. The Decennial Census, conducted every ten years, 
        focuses on counting the total population to determine congressional representation and federal funding allocation, 
        asking basic questions about household composition, housing type, ownership, and basic demographic information. 
        In contrast, the ACS is an annual survey that provides a detailed look at the social, economic, housing, and demographic characteristics of the population, 
        including information on education, employment, income, commute times, disability status, and more. 
        While the Decennial Census ensures an accurate population count for political and funding purposes, 
        the ACS offers insights into how people live and work, informing policy and planning decisions.
        
        Refer to the Census website https://www.census.gov/data.html
        for information on what these surveys contain.
    """)
    expander2 = st.expander("**Behind the Scenes**")
    expander2.write("""
        How does this work?
        
        Your natural language query is processed using LangChain to customize OpenAI's GPT 3.5 Turbo Model. The query goes through the following steps:

        The model identifies the question, relevant geographical area, variables, and Census datasets relevant to the natural language query.
        RAG systems based on GPT 3.5 Turbo identify what datasets and variables are most relevant to answering the question.
        A Census API call is created to get the data based on the most relevant parameters were identified via the RAG.
        The data goes through some simple python analysis which is displayed along with a GPT based description of the analysis results.      
    """)
    expander3 = st.expander("**What Can I Ask this Bot?**")
    expander3.write("""
        First, read through the description above of the Census and ACS datasets.
        Anything you can think of within the scope of those datasets is fair game to ask!
        
        If the model can find relevant data, it will give you the best response it can.
        
        If the output doesn't make sense or nothing is returned, try tweaking your
        question and asking again. The GPT model is doing its best, but you may have
        to try a few times to get sensible output in some cases!
        
        Question Examples: How many men are under 45 in
        Connectict? How many people commute to work via public transit
        in New York versus Los Angeles?""")
    expander4 = st.expander("**Data Privacy**")
    expander4.write("""
        None of your data that you enter is saved.
        
        Your API Keys and your queries will not be stored anywhere in this application. The only
        data we store is the embeddings of Census Metdata.
    """)


# main page content allowing for queries and data display
st.title("Census Query Bot")
input = st.text_input("Ask the Census Bot what you want to know from Census Data!")

with st.container():
    @st.cache_data
    def run(input):
        query = q.Query(q.api_access_url, q.variables, q.geography_fips)
        df = query.format_data()
        st.dataframe(df)
        explanation = query.explanation()
        st.text(explanation)
        analysis = a.Analysis(query).prompt()
        st.text(analysis)
        print(input)
        return None
    run('input goes here')

    
footer="""<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Abe Burton and Divij Sinha 2024</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
