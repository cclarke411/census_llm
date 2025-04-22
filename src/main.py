import streamlit as st
import pandas as pd

import os
from dotenv import load_dotenv

from census_server.chains import (
    SourceChain,
    SourceRAG,
    VariableTreeChain,
    RephraseChain,
    GeographyRAG,
    CensusQuery,
    AnalysisChain,
)

load_dotenv()
# check for API Keys:
open_ai_key_flag = False
if "OPENAI_API_KEY" in os.environ:
    open_ai_key_flag = True

census_key_flag = False
if "CENSUS_API_KEY" in os.environ:
    census_key_flag = True


def geo_lookup(geos):
    """Formats identified geographies from GeographyRAG into a dataframe for
    display to the user"""
    with open("/Users/clydeclarke/Documents/census_llm/src/census_server/data/fips/national_state2020.txt", "r") as file:
        states = file.readlines()
    state_rows = [state.replace("\n", "").split("|") for state in states]
    state_df = pd.DataFrame(state_rows[1:], columns=state_rows[0])

    with open("/Users/clydeclarke/Documents/census_llm/src/census_server/data/fips/national_county2020.txt", "r") as file:
        counties = file.readlines()
    county_rows = [county.replace("\n", "").split("|") for county in counties]
    county_df = pd.DataFrame(county_rows[1:], columns=county_rows[0])

    res = []
    for geo in geos:
        if isinstance(geo["state"], list):
            r = state_df.loc[state_df.STATEFP.isin(geo["state"])]
            res.append(r.loc[:, ["STATE", "STATE_NAME", "STATEFP"]])
        else:
            r = county_df.loc[
                (county_df.STATEFP == geo["state"])
                & (county_df.COUNTYFP.isin(geo["county"]))
            ]
            res.append(r.loc[:, ["STATE", "STATEFP", "COUNTYNAME", "COUNTYFP"]])
    return pd.concat(res).fillna("")


def process_geos(geos):
    """Formats identified geographies from GeographyRAG to a format usable by
    CensusQuery"""
    states_only = {"state": []}
    fixed_geos = {}
    for geo in geos:
        if "county" in geo:
            if geo["state"] in fixed_geos:
                fixed_geos[geo["state"]]["county"].append(geo["county"])
            else:
                fixed_geos[geo["state"]] = {
                    "state": geo["state"],
                    "county": [geo["county"]],
                }
        else:
            states_only["state"].append(geo["state"])
    if len(states_only["state"]) > 0:
        final_geos = [states_only]
    else:
        final_geos = []
    for _, geo in fixed_geos.items():
        final_geos.append(geo)
    return final_geos


@st.cache_data
def run(query: str, open_ai_key: str, census_key: str) -> None:
    """Run streamlit site with calls to server

    Args:
        query (str): user query
        open_ai_key (str): llm api key
        census_key (str): census api key

    Returns: None
    """
    # rephrase question
    rephrase_chain = RephraseChain(open_ai_key)
    ans = rephrase_chain.invoke(query)

    st.write("**Rephrased Question**")
    st.write(ans["rephrased_question"])

    # indentify sources
    source_chain = SourceChain(open_ai_key)
    ans = source_chain.invoke(ans["rephrased_question"])

    source_rag = SourceRAG(open_ai_key)
    doc = source_rag.invoke(query, ans["variables"], ans["relevant_dataset"])
    access_link = doc.metadata["distribution"][0]["accessURL"]

    st.write("**Identified Data Source:**")
    st.write(doc.page_content)
    st.write("**Searching for Variables:** ", ans["variables"])

    st.divider()

    # identify relevant variables in sources
    variable_rag = VariableTreeChain(doc.metadata["c_variablesLink"], open_ai_key)
    vars = variable_rag.invoke(query, ans["variables"], ans["relevant_dataset"])
    st.write("**Variables Found:**")
    var_df = pd.DataFrame(
        [(var["code"], var["concept"], var["label"]) for var in vars.values()]
    )
    var_df.columns = ["code", "concept", "label"]
    st.write(var_df)

    st.divider()

    # identify relevant geographies to the query
    geos = []
    geo_rag = GeographyRAG(open_ai_key)
    for geo in ans["geography"]:
        res = geo_rag.invoke(geo)
        geos.append(res)
    geos = process_geos(geos)
    st.write("**Geographies Identified:**")
    st.write(geo_lookup(geos))

    st.divider()

    # build and run census api call
    st.write("**Pulling Data...**")
    dfs = []
    for geo in geos:
        query = CensusQuery(
            api_access_url=access_link,
            variables=vars,
            geography=geo,
            census_key=census_key,
        )
        df = query.get_data()
        dfs.append(df)
    df = pd.concat(dfs)
    st.dataframe(df)

    st.divider()

    # perform basic analysis of the data
    st.write("**Data Analysis:**")
    analysis = AnalysisChain(df, vars, open_ai_key)
    res = analysis.invoke()
    st.write(res)

    st.divider()

    return None


# sidebar with FAQ
load_dotenv()
with st.sidebar:
    global user_open_ai_key, user_census_key
    st.markdown("**Enter your OpenAI API Key**")
    user_open_ai_key = st.text_input("For accessing GPT")
    st.write(
        "*Get your OpenAI key here* :point_down: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key"
    )
    st.markdown("**Enter your Census API Key**")
    user_census_key = st.text_input("For accessing Census Data")
    st.write(
        "*Get your Census API key here* :point_down: https://api.census.gov/data/key_signup.html"
    )

    if user_open_ai_key == "" and not open_ai_key_flag:
        st.error(f"Please enter your OpenAI API Key!", icon="🚨")
    if user_census_key == "" and not census_key_flag:
        st.error(f"Please enter your Census API Key!", icon="🚨")

    expander1 = st.expander("**Data Source Information**")
    expander1.write(
        """
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
    """
    )
    expander2 = st.expander("**Behind the Scenes**")
    expander2.write(
        """
        How does this work?
        
        Your natural language query is processed using LangChain to customize OpenAI's GPT 3.5 Turbo Model. The query goes through the following steps:

        The model identifies the question, relevant geographical area, variables, and Census datasets relevant to the natural language query.
        RAG systems based on GPT 3.5 Turbo identify what datasets and variables are most relevant to answering the question.
        A Census API call is created to get the data based on the most relevant parameters were identified via the RAG.
        The data goes through some simple python analysis which is displayed along with a GPT based description of the analysis results.      
    """
    )
    expander3 = st.expander("**What Can I Ask this Bot?**")
    expander3.write(
        """
        First, read through the description above of the Census and ACS datasets.
        Anything you can think of within the scope of those datasets is fair game to ask!
        
        We've restricted the API to look for states and counties, so for best results,
        make sure your questions are either about the US, or are on the state or county level.
        City or Census Tract level questions will have worse output or just return statistics for
        the counties they are in.
        
        If the model can find relevant data, it will give you the best response it can.
        
        If the output doesn't make sense or nothing is returned, try tweaking your
        question and asking again. The GPT model is doing its best, but you may have
        to try a few times to get sensible output in some cases!
        
        Think of this site like an aid for learning about the Census without 
        navigating the huge Census website or learning to write API calls. You may
        have to workshop your questions for best results, but this will have you
        accessing answers to any Census/ACS questions fast!  
      """
    )
    expander4 = st.expander("**Data Privacy**")
    expander4.write(
        """
        None of your data that you enter is saved.
        
        Your API Keys and your queries will not be stored anywhere in this application. The only
        data we store is the embeddings of Census Metdata.
    """
    )


# main page content allowing for queries and data display
st.title("Census Query Bot")

col1, col2 = st.columns([1, 1])

with col1:
    input = st.text_input("Ask the Bot what you want to know from Census Data!")
with col2:
    option = st.selectbox(
        label="Need Ideas? Pick from here:",
        options=(
            "How many men are under 45 in Utah?",
            "How many people commute via public transit in New Jersey?",
            "What percent of people work in arts and entertainment in California?",
            "What percent of  Maricopa County, Arizona is Hispanic?",
        ),
        index=None,
    )

if option is not None:
    input = option

st.divider()

with st.container():
    if input != "":
        if user_open_ai_key != "":
            open_ai_key = user_open_ai_key
        elif open_ai_key_flag:
            open_ai_key = os.environ["OPENAI_API_KEY"]

        if user_census_key != "":
            census_key = user_census_key
        elif census_key_flag:
            census_key = os.environ["CENSUS_API_KEY"]
        else:
            census_key = ""

        st.header("Results:")

        run(input, open_ai_key, census_key)


footer = """<style>
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
st.markdown(footer, unsafe_allow_html=True)
