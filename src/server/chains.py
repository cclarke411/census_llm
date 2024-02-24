from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

import json
import os
from dotenv import load_dotenv
from pathlib import Path
import requests

load_dotenv()

script_path = Path(__file__).resolve()
script_dir = script_path.parent
HEAD = "http://api.census.gov/data/"


def get_data(file_path, key, keep):
    with open(file_path, "r") as file:
        data = json.load(file)
    datasets_to_keep = []
    for dataset in data[key]:
        if isinstance(dataset, str):
            code = dataset
            dataset = data[key][code]
            dataset["code"] = code
        dataset_strings = []
        for keep_var in keep:
            dataset_strings.append(dataset.get(keep_var, ""))
        dataset_tuple = ("---".join(dataset_strings), dataset)
        datasets_to_keep.append(dataset_tuple)
    return datasets_to_keep


def save_docembedding(embeddings_folder_path, datasets):
    data, metadata = zip(*datasets)
    splitter = CharacterTextSplitter(chunk_size=2750, chunk_overlap=0)
    xml_docs = splitter.create_documents(data, metadata)
    docembeddings = FAISS.from_documents(xml_docs, OpenAIEmbeddings())
    docembeddings.save_local(embeddings_folder_path)


class SourceChain:
    def __init__(self) -> None:
        self.template = """Split the question into three parts, the geographic region(s), 
        the general category(s) of the variables mentioned, and the relevant dataset name.
        Only use language from the question.

        If you don't know the answer, just say that you don't know, don't try to 
        make up an answer. Only use the provided context to answer.

        This should be in the following format
        Question: [question]
        Geography: [geography]
        Variable: [variable category(s)]
        Dataset: [dataset name]

        -----
        
        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.output_parser = RegexParser(
            regex=r"Geography:(.*?)\nVariable: (.*)\nDataset:(.*)",
            output_keys=["geography", "variable", "dataset"],
        )
        self.chain = self.prompt | self.model | self.output_parser

    def invoke(self, question: str):
        return self.chain.invoke({"question": question})


class SourceRAG:
    def __init__(self) -> None:
        self.template = """Given the question, the identified categories, and the approximate dataset, 
        score how likely the dataset in the metadata is to answer the question 

        Only use the provided metadata to answer.
        If your score is more than 0, set the Answer to True, otherwise set the Answer to False 

        This should be in the following format
        Question: [question]
        Identified Categories: [categories]
        Dataset: [dataset name]
        Answer: [answer]
        Score: [score between 0 and 100]
        
        -----
        
        {context}

        Question: {question}
        Identified Categories: {categories}
        Dataset: {dataset}
        Answer: """

        output_parser = RegexParser(
            regex=r"(.*?)\n\s+Score: (.*)",
            output_keys=["answer", "score"],
        )
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question", "categories", "dataset"],
            output_parser=output_parser,
        )
        self.docembeddings = self.get_api_discovery_docembedding()
        self.chain = load_qa_chain(
            OpenAI(temperature=0),
            chain_type="map_rerank",
            return_intermediate_steps=True,
            prompt=self.prompt,
        )

    def invoke(self, question, categories, dataset):
        self.relevant_chunks = self.docembeddings.similarity_search_with_score(
            "  ".join([question, categories, dataset]), k=3
        )
        chunks, scores = zip(*self.relevant_chunks)
        self.results = self.chain.invoke(
            {
                "input_documents": chunks,
                "question": question,
                "categories": categories,
                "dataset": dataset,
            }
        )
        return self.results

    def get_api_discovery_data(self, vintage=2020):
        file_path = script_dir / Path("data/api_discovery.json")
        key = "dataset"
        keep = ["title", "description"]
        datasets = get_data(file_path, key, keep)
        datasets_to_keep = []
        for data_string, data_dict in datasets:
            if data_dict.get("c_vintage", None) == vintage:
                datasets_to_keep.append((data_string, data_dict))
        return datasets_to_keep

    def get_api_discovery_docembedding(self):
        api_discovery_path = (
            script_dir / "faiss_index" / "llm_faiss_index_api_discovery"
        )
        if not os.path.exists(api_discovery_path):
            datasets = self.get_api_discovery_data()
            save_docembedding(api_discovery_path, datasets)
        docembeddings = FAISS.load_local(api_discovery_path, OpenAIEmbeddings())
        return docembeddings


class VariableRAG:
    def __init__(self, variable_url) -> None:
        self.save_variables(variable_url)

        self.template = """Given the question, the identified categories, and the dataset, 
        score how likely the given label and concept is to answer the question 

        Only use the provided metadata to answer.
        If your score is more than 0, set the Answer to the given label and concept, otherwise set the Answer to False 

        This should be in the following format
        Question: [question]
        Identified Categories: [categories]
        Dataset: [dataset name]
        Answer: [answer]
        Score: [score between 0 and 100]
        
        -----
        
        Label and Concept: {context}

        Question: {question}
        Identified Categories: {categories}
        Dataset: {dataset}
        Answer: """

        output_parser = RegexParser(
            regex=r"(.*?)\n\s+Score: (.*)",
            output_keys=["answer", "score"],
        )
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question", "categories", "dataset"],
            output_parser=output_parser,
        )
        self.docembeddings = self.get_variable_docembedding()
        self.chain = load_qa_chain(
            OpenAI(temperature=0),
            chain_type="map_rerank",
            return_intermediate_steps=True,
            prompt=self.prompt,
        )

    def save_variables(self, variable_url):
        self.url = variable_url
        self.file_path = Path(
            script_dir / "data" / self.url.replace(HEAD, "").replace("/", "_")
        )
        if not os.path.exists(self.file_path):
            resp = requests.get(self.url)
            with open(self.file_path, "wb") as file:
                file.write(resp.content)

    def invoke(self, question, categories, dataset):
        self.relevant_chunks = self.docembeddings.similarity_search_with_score(
            "  ".join([question, categories, dataset]), k=3
        )
        chunks, scores = zip(*self.relevant_chunks)
        self.results = self.chain.invoke(
            {
                "input_documents": chunks,
                "question": question,
                "categories": categories,
                "dataset": dataset,
            }
        )
        return self.results

    def get_variable_data(self):
        key = "variables"
        keep = ["label", "concept"]
        datasets = get_data(self.file_path, key, keep)
        return datasets

    def get_variable_docembedding(self):
        docembedding_path = (
            script_dir
            / "faiss_index"
            / f"llm_faiss_index_{self.url.replace(HEAD, '').replace('/', '_')}"
        )
        if not os.path.exists(docembedding_path):
            datasets = self.get_variable_data()
            save_docembedding(docembedding_path, datasets)
        docembeddings = FAISS.load_local(docembedding_path, OpenAIEmbeddings())
        return docembeddings
