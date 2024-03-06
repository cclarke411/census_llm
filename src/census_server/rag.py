# %%
"""
Author: Abe Burton, Divij Sinha

This file builds a rag using census metadata that takes in natural language
query input and outputs a dictionary with the relevant dataset information 
necessary to create a structured query for that data
"""
from chains import SourceChain, SourceRAG, VariableRAG, RephraseChain, GeographyRAG
from pathlib import Path
import json

# %%
script_path = Path(__file__).resolve()
script_dir = script_path.parent

query = (
    "How many people are under the age of 45 according to the Decennial Census in Utah?"
)

rc = RephraseChain()
ans = rc.invoke(query)
print(ans)
sc = SourceChain()
ans = sc.invoke(ans["rephrased_question"])
print(ans)
sr = SourceRAG()
doc = sr.invoke(query, ans["variables"], ans["relevant_dataset"])
print(doc.page_content)
access_link = doc.metadata["distribution"][0]["accessURL"]
vr = VariableRAG(doc.metadata["c_variablesLink"])
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
