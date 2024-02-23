"""
Author: Abe Burton

This file builds a rag using census metadata that takes in natural language
query input and outputs a dictionary with the relevant dataset information 
necessary to create a structured query for that data
"""
import os
import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.output_parsers import RegexParser
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv

load_dotenv()

query = "How many men are under 45 in Utah according to the 2020 Census?"

def read_metadata():

    # get json data
    json_loader = JSONLoader(
    file_path='./variables_source.json',
    jq_schema='.variables[]',
    text_content=False)
    json_documents = json_loader.load()
    
    general_loader = DirectoryLoader('./docs/', glob="*")
    docs = general_loader.load()
    
    docs.extend(json_documents)
    
    python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0)
    python_docs = python_splitter.split_documents(docs)
    docembeddings = FAISS.from_documents(python_docs, OpenAIEmbeddings())
    
    docembeddings.save_local("llm_faiss_index")
    docembeddings = FAISS.load_local("llm_faiss_index",OpenAIEmbeddings())

    return docembeddings


# get document embeddings
docembeddings = read_metadata()

# create the prompt
template = """ Split the query into three parts: the geographic region, the variable(s) concepts, and the source that are
relevant to the query. 

Forget everything you know about the Census and the American Community Survey.
Only use the given context to answer the question.

Use the only the following pieces of context from the 2020 Census to identify which fips code is relevant to the identified geographic region.
Use the only the following pieces of context from the 2020 Census to identify what the variable(s) key(s) is for the identified variable(s) concepts.
Closely match the identified concept to the concept key of each variable.
Match the source key of the identified variable(s) to the source specified in the query.
If the source you cite is not in the context, penalize your score by -50.

If you don't know the answer, just say that you don't know, don't try to make up an answer. Only use the provided context to answer.

This should be in the following format:

Question: [question here]
Exact Answer: 'variable':[variable key here],'fips':[fips code here], Source: [Tell me where you found the variable key]
Score: [score between 0 and 100]

{context}

Question: {question}

Exact Answer:"""

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    output_parser=output_parser
)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)

def getanswer(query):
    relevant_chunks = docembeddings.similarity_search_with_score(query,k=2)
    chunk_docs=[]
    for chunk in relevant_chunks:
        chunk_docs.append(chunk[0])
    results = chain({"input_documents": chunk_docs, "question": query})
    text_reference=""
    for i in range(len(results["input_documents"])):
        text_reference+=results["input_documents"][i].page_content
    output={"Answer":results["output_text"],"Reference":text_reference}
    return output

output = getanswer(query)
print(output)

#%%

# import json

# with open("variables.json", "r") as f:
#     d = json.load(f)

# source_d = {"variables":{}}   
# for key in d["variables"].keys():
#     cur_d = d["variables"][key]
#     cur_d["Source"] = "American Decennial Census, 2020"
#     source_d["variables"][key] = cur_d
    
# with open("variables_source.json", "w") as f:
#     json.dump(source_d, f)

# %%
