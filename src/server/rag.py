# %%
"""
Author: Abe Burton, Divij Sinha

This file builds a rag using census metadata that takes in natural language
query input and outputs a dictionary with the relevant dataset information 
necessary to create a structured query for that data
"""
from chains import SourceChain, SourceRAG
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

script_path = Path(__file__).resolve()
script_dir = script_path.parent

query = "How many men are under 45 in Utah according to the 2020 Decennial Census?"

sc = SourceChain()
ans = sc.invoke(query)

sr = SourceRAG()
res = sr.invoke(query, ans["variable"], ans["dataset"])

print(res["output_text"])

for doc in res["input_documents"]:
    print("---")
    print(doc)
    print("---", end="\n\n")
