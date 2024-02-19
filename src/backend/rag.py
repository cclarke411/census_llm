"""
Author: Abe Burton

This file builds a rag using census metadata that takes in natural language
query input and outputs a dictionary with the relevant dataset information 
necessary to create a structured query for that data
"""
import os
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.output_parsers import RegexParser
from dotenv import load_dotenv

load_dotenv()

# query = natural language query from frontend

def read_metadata():
    chunk_size_value = 1000
    chunk_overlap=100
    
    # get html data
    html_loader = BSHTMLLoader("example_data/fake-content.html")
    html_documents = html_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap,length_function=len)
    texts = text_splitter.split_documents(html_documents)
    
    # get json data https://python.langchain.com/docs/modules/data_connection/document_loaders/json
    json_loader = JSONLoader(
    file_path='./example_data/facebook_chat.json',
    jq_schema='.messages[].content',
    text_content=False)
    json_documents = json_loader.load()
    
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/HTML_header_metadata
    # md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=chunk_size_value, chunk_overlap=chunk_overlap, length_function=len)
    # notebook_texts = md_splitter.split_documents(notebook_documents)
    
    # get embeddings for all the documents
    all_texts = json_texts + html_texts
    
    docembeddings = FAISS.from_documents(all_texts, OpenAIEmbeddings())
    docembeddings.save_local("llm_faiss_index")
    docembeddings = FAISS.load_local("llm_faiss_index",OpenAIEmbeddings())
    
    return docembeddings


# get document embeddings
docembeddings = read_metadata()

# create the prompt
template = """Use the following pieces of context to identify which datasets or variables could answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

{context}

Question: {question}

Helpful Answer:"""

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

#output = getanswer(query)