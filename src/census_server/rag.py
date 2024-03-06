# %%
"""
Author: Abe Burton, Divij Sinha

This file builds a rag using census metadata that takes in natural language
query input and outputs a dictionary with the relevant dataset information 
necessary to create a structured query for that data
"""
from chains import (
    SourceChain,
    SourceRAG,
    VariableTreeChain,
    RephraseChain,
    GeographyRAG,
)
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
# %%
script_path = Path(__file__).resolve()
script_dir = script_path.parent
# %%
query = (
    "How many men are under the age of 45 according to the Decennial Census in Utah?"
)

rc = RephraseChain(os.environ["OPENAI_API_KEY"])
ans = rc.invoke(query)
print(ans)
sc = SourceChain(os.environ["OPENAI_API_KEY"])
ans = sc.invoke(ans["rephrased_question"])
print(ans)
sr = SourceRAG(os.environ["OPENAI_API_KEY"])
doc = sr.invoke(query, ans["variables"], ans["relevant_dataset"])
print(doc.page_content)
access_link = doc.metadata["distribution"][0]["accessURL"]
# %%
vr = VariableTreeChain(doc.metadata["c_variablesLink"], os.environ["OPENAI_API_KEY"])
# %%
res = vr.invoke(query, ans["variables"], ans["relevant_dataset"])
print(res)
access_variable_code = res.metadata["code"]
geos = []
g = GeographyRAG()
for geo in ans["geography"]:
    res = g.invoke(geo)
    geos.append(res)
print(geos)
# %%
geos = []
geo_rag = GeographyRAG(os.environ["OPENAI_API_KEY"])
for geo in ["Utah", "NY", "Chicago", "baltimore"]:
    res = geo_rag.invoke(geo)
    geos.append(res)

states_only = {"state": []}
fixed_geos = {}
for geo in geos:
    if "county" in geo:
        if geo["state"] in fixed_geos:
            fixed_geos[geo["state"]].append(geo["county"])
        else:
            fixed_geos[geo["state"]] = [geo["county"]]
    else:
        states_only["state"].append(geo["state"])
# %%
