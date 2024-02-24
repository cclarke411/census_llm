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

load_dotenv()

script_path = Path(__file__).resolve()
script_dir = script_path.parent


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
            question, k=3
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
        if not isinstance(vintage, int):
            vintage = int(vintage)

        file_path = script_dir / Path("data/api_discovery.json")
        with open(file_path, "r") as file:
            api_list = json.load(file)

        keep = ["title", "description"]
        datasets_to_keep = []
        for dataset in api_list["dataset"]:
            if dataset.get("c_vintage", None) == vintage:
                dataset_strings = []
                for keep_var in keep:
                    dataset_strings.append(dataset[keep_var])
                dataset_tuple = ("---".join(dataset_strings), dataset)
                datasets_to_keep.append(dataset_tuple)
        return datasets_to_keep

    def get_api_discovery_docembedding(self):
        api_discovery_path = script_dir / "llm_faiss_index_api_discovery"
        if not os.path.exists(api_discovery_path):
            vintage = "2020"
            datasets = self.get_api_discovery_data(vintage)
            data, metadata = zip(*datasets)
            splitter = CharacterTextSplitter(chunk_size=2750, chunk_overlap=0)
            xml_docs = splitter.create_documents(data, metadata)
            docembeddings = FAISS.from_documents(xml_docs, OpenAIEmbeddings())
            docembeddings.save_local(api_discovery_path)
        docembeddings = FAISS.load_local(api_discovery_path, OpenAIEmbeddings())

        return docembeddings
