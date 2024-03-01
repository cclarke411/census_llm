# %%
"""
Author: Abe Burton, Divij Sinha

This file builds a rag using census metadata that takes in natural language
query input and outputs a dictionary with the relevant dataset information 
necessary to create a structured query for that data
"""
from chains import SourceChain, SourceRAG, VariableRAG, RephraseChain
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
# %%
script_path = Path(__file__).resolve()
script_dir = script_path.parent

query = "What is the population count of males aged 0 to 45 in Utah, as per the 2020 Decennial Census data?"

rc = RephraseChain()
ans = rc.invoke(query)
print(ans)
sc = SourceChain()
ans = sc.invoke(ans["rephrased_question"])
print(ans)
sr = SourceRAG()
doc = sr.invoke(query, ans["variable"], ans["dataset"])
print(doc)
vr = VariableRAG(doc.metadata["c_variablesLink"])
res = vr.invoke(query, ans["variable"], ans["dataset"])
print(res)
# %%
