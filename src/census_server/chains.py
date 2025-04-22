from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

import pandas as pd
from pathlib import Path
from operator import itemgetter

import requests
import json
import re
import io
import os

script_path = Path(__file__).resolve()
script_dir = script_path.parent
HEAD = "http://api.census.gov/data/"


def format_docs(docs):
    formatted_str = ""
    for idx, doc in enumerate(docs):
        formatted_str += f"DOCUMENT {idx+1}\nCONTENT: {doc.page_content}\n\n\n\n"
    return formatted_str


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
        dataset_tuple = ("---".join(dataset_strings).replace(":", "").strip(), dataset)
        datasets_to_keep.append(dataset_tuple)
    return datasets_to_keep


def save_docembedding(embeddings_folder_path, datasets, open_ai_key, separator="\n\n"):
    data, metadata = zip(*datasets)
    splitter = CharacterTextSplitter(
        chunk_size=2750, chunk_overlap=0, separator=separator
    )
    xml_docs = splitter.create_documents(data, metadata)
    docembeddings = FAISS.from_documents(
        xml_docs, OpenAIEmbeddings(api_key=open_ai_key)
    )
    docembeddings.save_local(embeddings_folder_path)


class RephraseChain:
    """Rephrases a user question in a more parseable way"""

    def __init__(self, open_ai_key) -> None:
        self.template = """Rephrase the following question, if needed, to be more helpful in identifying variables
        
        Format it as a json with the key rephrased_question
        -----
        
        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, api_key=open_ai_key
        )
        self.output_parser = SimpleJsonOutputParser()
        self.chain = self.prompt | self.model | self.output_parser

    def invoke(self, question: str):
        return self.chain.invoke({"question": question})


class SourceChain:
    """Identifies geographies, datasets, and variables that are referenced in a
    user question
    """

    def __init__(self, open_ai_key) -> None:
        self.template = """Split the question into three parts, the geographic region(s), 
        the core concepts of the variable(s) mentioned, and the relevant dataset name.
        Only use language from the question.

        If you don't know the answer, just say that you don't know, don't try to 
        make up an answer. Only use the provided context to answer.

        Return a json with the keys "geography", "relevant_dataset", and "variables". 
        the value for the key "geography" should be a list. If no place is mentioned set it to an empty list.
        the value for the key "relevant_dataset" should be a string. If no dataset is mentioned, set it to NA
        the value for the key "variables" should be a list of the core concepts of the variable(s) in the question.

        This should be in the following format
        Question: [question]
        [answer json in specified format goes here]

        -----
        
        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, api_key=open_ai_key
        )
        self.output_parser = SimpleJsonOutputParser()
        self.chain = self.prompt | self.model | self.output_parser

    def invoke(self, question: str):
        return self.chain.invoke({"question": question})


class SourceRAG:
    """RAG identifies the most relevant source dataset in Census Metadata based
    on previous functions identified question, categories and datasets
    """

    def __init__(self, open_ai_key) -> None:
        self.template = """
        You are a helpful Census Agent. Only use the provided metadata to answer.
        You are given the question, the identified categories, and the approximate dataset, 
        Your task is to choose the best DOCUMENT from the given list of DOCUMENTS. 
        Identify the DOCUMENT that best matches the Question, Identified Queries, and the approximate Dataset.

        Set Answer equal to a json with the keys "doc_title" and "doc_content"

        List of Datasets:
        {context}

        Question: {question}
        Identified Categories: {categories}
        Dataset: {dataset}
        Answer: """

        self.docembeddings = self.get_api_discovery_docembedding(open_ai_key)
        self.docretriever = self.docembeddings.as_retriever(
            search_kwargs={"k": 3},
        )
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question", "categories", "dataset"],
        )
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, api_key=open_ai_key
        )
        self.output_parser = SimpleJsonOutputParser()

    def invoke(self, question, categories, dataset):
        self.chain = (
            {
                "context": itemgetter("question") | self.docretriever | format_docs,
                "question": itemgetter("question"),
                "categories": itemgetter("categories"),
                "dataset": itemgetter("dataset"),
            }
            | self.prompt
            | self.model
            | self.output_parser
        )

        self.results = self.chain.invoke(
            {
                "question": question,
                "categories": categories,
                "dataset": dataset,
            }
        )
        res_docs = self.docretriever.get_relevant_documents(self.results["doc_content"])
        self.res_doc = res_docs[0]

        return self.res_doc

    def get_api_discovery_data(self):
        file_path = script_dir / Path("data/api_discovery.json")
        key = "dataset"
        keep = ["title", "description"]
        datasets = get_data(file_path, key, keep)
        return datasets

    def get_api_discovery_docembedding(self, open_ai_key):
        api_discovery_path = (
            script_dir / "data" / "faiss_index" / "llm_faiss_index_api_discovery"
        )
        if not os.path.exists(api_discovery_path):
            datasets = self.get_api_discovery_data()
            save_docembedding(api_discovery_path, datasets, open_ai_key)
        docembeddings = FAISS.load_local(
            api_discovery_path, OpenAIEmbeddings(api_key=open_ai_key),
            allow_dangerous_deserialization=True

        )
        return docembeddings


class VariableTreeChain:
    """Given a Census source dataset, returns relevant census variables scored
    above a threshold by constructing and traversing a tree using a RAG
    """

    def __init__(self, variable_url, open_ai_key) -> None:
        self.save_variables(variable_url)

        self.template = """
        ##CONTEXT:##
        You are a helpful Census Agent. Only use the provided metadata to answer.
        You are given the question entered by the user, the categories relevant to the question, and the dataset.

        ##OBJECTIVE:##
        You are trying to identify all the VARIABLE relevant to the question.
        If the VARIABLE directly answer the question, choose all the VARIABLE.
        If the VARIABLE do not answer the question, or answer a different question, do not choose the VARIABLE.
        Do not choose a VARIABLE that is only tangentially related to the question.
        Do not choose a VARIABLE that is more detailed than needed. Stick to the question.
        If the question does not have a race mentioned, do not choose a VARIABLE that has a race mentioned in the VARIABLE.
        If the question does have a race mentioned, only choose the VARIABLE that has the race mentioned in the VARIABLE corresponding to the question.

        ##INFORMATION PROVIDED:##
        You are also given a list of VARIABLE. EACH VARIABLE REPRESENTS A VARIABLE or a VARIABLE STEM.
        Your task is to choose the best one or multiple VARIABLE from the given list of VARIABLE.

        Remember, each VARIABLE could be just the partial variable.   
        Remember, each VARIABLE could be the full variable.   
        Remember, more than one VARIABLE could be accurate.   
        Do not give multiples if they represent the general same theme. 
        Do give multiples if they are different and add variety. 

        Set Answer equal to a json with the keys "var_content", "var_scores". 
        Set the values of both keys to lists.
        For "var_content", set each element of the list in the value equal to the DESCRIPTION of the VARIABLE.
        For "var_score", set each element of the list in the value equal to a score in the range 0-100 of how perfectly the VARIABLE is to answer the question.
        RETURN THE DESCRIPTION AS IS, DO NOT CHANGE ANYTHING.
        
        INFORMATION:::
        List of Variable: 
        {context}

        Question: {question}
        Identified Categories: {categories}
        Dataset: {dataset}
        Answer: """

        self.open_ai_key = open_ai_key
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question", "categories", "dataset"],
        )
        self.var_tree = self.get_variable_data()
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, api_key=open_ai_key
        )
        self.output_parser = SimpleJsonOutputParser()
        self.chain = self.prompt | self.model | self.output_parser

    def save_variables(self, variable_url):
        self.url = variable_url
        self.file_path = Path(
            script_dir / "data" / self.url.replace(HEAD, "").replace("/", "_")
        )
        if not os.path.exists(self.file_path):
            resp = requests.get(self.url)
            with open(self.file_path, "wb") as file:
                file.write(resp.content)

    def rec_invoke(self, cur_tree):
        """Searches the variable tree. Only traverses to next depth if current
        level scored above a threshold. If level scored higher than threshold,
        and variable exists at leaf node, adds to result.
        """
        if len(cur_tree.children) == 0:
            self.results[cur_tree.dataset[1]["code"]] = cur_tree.dataset[1]
        elif len(cur_tree.children) == 1:
            next_tree = cur_tree.children[list(cur_tree.children.keys())[0]]
            self.rec_invoke(next_tree)
        else:
            formatted_variable_str = self.format_vars(cur_tree)
            splitter = CharacterTextSplitter(chunk_size=30_000)
            splitted_variable_str = splitter.split_text(formatted_variable_str)
            for var_str in splitted_variable_str:
                print(var_str)
                cur_results = self.chain.invoke(
                    {
                        "context": var_str,
                        "question": self.question,
                        "categories": self.categories,
                        "dataset": self.dataset,
                    }
                )
                print(cur_results)
                if isinstance(cur_results["var_content"], list):
                    for idx, next_child_result in enumerate(cur_results["var_content"]):
                        if int(cur_results["var_scores"][idx]) > 30:
                            next_child = next_child_result.split("---")[0].split("!!")[
                                -1
                            ]
                            next_tree = cur_tree.children[next_child]
                            self.rec_invoke(next_tree)
                else:
                    if int(cur_results["var_scores"]) > 30:
                        next_child = (
                            cur_results["var_content"].split("---")[0].split("!!")[-1]
                        )
                        next_tree = cur_tree.children[next_child]
                        self.rec_invoke(next_tree)

    def invoke(self, question, categories, dataset):
        cur_tree = self.var_tree
        self.question = question
        self.categories = ", ".join(categories)
        self.dataset = dataset
        self.results = {}
        self.rec_invoke(cur_tree)
        return self.results

    def get_variable_data(self):
        key = "variables"
        keep = ["label", "concept"]
        datasets = get_data(self.file_path, key, keep)
        v = VarTree()
        for data, metadata in datasets:
            if metadata["label"] not in [
                "Geography",
                "Census API FIPS 'for' clause",
                "Census API FIPS 'in' clause",
                "Uniform Census Geography Identifier clause",
                "Summary Level code",
                "GEO_ID Component",
            ]:
                try:
                    branch = (
                        metadata["label"]
                        .strip()
                        .replace(":", "")
                        .replace("--", "")
                        .strip()
                    )
                    branch = re.sub("^!!", "", branch).strip().split("!!")
                    branch = [metadata["concept"]] + branch
                    v.append(branch, (data, metadata))
                except:
                    pass
        return v

    def format_vars(self, v):
        formatted_str = ""
        for idx, (key, item) in enumerate(v.children.items()):
            item_str = f"\n\nVARIABLE {idx+1}\nDESCRIPTION:"
            if item.dataset is not None:
                item_str += item.dataset[0]
            else:
                item_str += key
            formatted_str += item_str
        return formatted_str


class VarTree:
    """Variable Tree that creates hierarchical variable category structure
    exploiting the tree-like structure of the census. Uses concept as root to
    branch out on.
    """

    def __init__(self) -> None:
        self.children = {}
        self.dataset = None

    def append(self, branch, label_dataset):
        if len(branch) == 0:
            self.dataset = label_dataset
        elif branch[0] in self.children.keys():
            self.children[branch[0]].append(branch[1:], label_dataset)
        else:
            v = VarTree()
            v.append(branch[1:], label_dataset)
            self.children[branch[0]] = v


class GeographyRAG:
    """Class to identify and return most relevent county/state FIPS code identified in a query"""

    def __init__(self, open_ai_key) -> None:
        self.template = """
        As a Census Representative, your objective is to identify the accurate FIPS code 
        for only the place specified in the query, utilizing only the metadata provided.

        Here's your guide to handling the queries:

        First, determine the state that the queried place is referring to. 
        In the answer json, set the key "state" equal to the 2-digit FIPS code for the state as requested. 
        In the answer json, set the key "type" equal to the string "state"

        If the queried place is only referring to a state, stop here and return the resulting json.
        Remember, if the query exclusively mentions a state, do not use county codes.

        Second, determine if the place in the query specifically refers to a county or county equivalent from the provided data.
        In the answer json, set the key "county" equal to the 3-digit FIPS code for the county or the county equivalent as requested. 
        In the answer json, set the key "type" equal to the string "county"

        If the queried place is only referring to a county or county equivalent, stop here and return the resulting json.

        Third, if you have determined that the queried place is not referring to a state or a county or county equivalent, 
        determine the most appropriate county for this place.
        In the answer json, set the key "state" equal to the 2-digit FIPS code for the state as requested. 
        In the answer json, set the key "county" equal to the 3-digit FIPS code for the county or the county equivalent as requested. 
        In the answer json, set the key "type" equal to the string "other"

        Return the jsom

        - **How to Respond:** Your response should consist of the numeric FIPS codes only, omitting the names. Structure your response as a JSON object with the keys "states", "counties", and "type". 
       
        **JSON FORMAT**
        key: value
        state: list of state
        county: list of counties
        type: string of type
        **END OF JSON FORMAT** 

        **Metadata Utilization:** Ensure all information is derived from the given metadata for accuracy.

        **Response Format:** Ensure your answer follows the specified JSON structure for clarity and precision.

        After you think of the answer, cross check whether anything in your answer is not corresponding to the Queried Place. Remove the offending parts.
        Do this until you are satisfied with your answer.
        
        ###QUERIED PLACE###
        {question}
        ###END OF QUERY###

        ###METADATA FOR CONTEXT###
        {context}
        ###END OF METADATA###
        
        Answer: """

        self.docembeddings = self.get_fips_docembedding(open_ai_key)
        self.docretriever = self.docembeddings.as_retriever(
            search_kwargs={"k": 3},
        )
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"],
        )
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0, api_key=open_ai_key
        )
        self.output_parser = SimpleJsonOutputParser()

    def invoke(self, question):
        self.chain = (
            {
                "context": itemgetter("question") | self.docretriever | format_docs,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.model
            | self.output_parser
        )

        self.results = self.chain.invoke({"question": question})
        return self.results

    def get_fips_data(self):
        file_path = script_dir / Path("data/fips/")
        datasets_w_meta = []
        for file_name in os.listdir(file_path):
            with open(file_path / file_name, "r") as f:
                datasets_w_meta.append((f.read(), {"file_name": file_name}))
        return datasets_w_meta

    def get_fips_docembedding(self, open_ai_key):
        fips_path = script_dir / "data" / "faiss_index" / "llm_faiss_index_fips"
        if not os.path.exists(fips_path):
            datasets = self.get_fips_data()
            save_docembedding(fips_path, datasets, open_ai_key, separator="\n")
        docembeddings = FAISS.load_local(
            fips_path, OpenAIEmbeddings(api_key=open_ai_key),
            allow_dangerous_deserialization=True

        )
        return docembeddings


class CensusQuery:
    """Builds a valid census query using identified parameters and api key"""

    def __init__(
        self, api_access_url: str, variables: dict, geography: dict, census_key
    ):
        self.census_key = census_key
        self.url = api_access_url
        self.variables = variables
        self.geographies = geography
        self.df = None

    def build_query(self):
        query = [self.url, "?get="]

        # specify dataset variables to extract
        vars = list(self.variables.keys())
        vars_csv = ",".join(vars)
        query.append(vars_csv)

        # specify geographies to include
        state, county = ("*", "*")
        if "state" in self.geographies:
            state = self.geographies["state"]
            if isinstance(state, list):
                state = ",".join(state)
        if "county" in self.geographies:
            county = self.geographies["county"]
            if isinstance(county, list):
                county = ",".join(county)
            geo_string = f"&for=county:{county}&in=state:{state}"
        else:
            geo_string = f"&for=state:{state}"

        # include api key at the end of the query
        query.append(geo_string)
        api_key = f"&key={self.census_key}"
        query.append(api_key)

        return "".join(query)

    def dl_data(self):
        try:
            api_url = self.build_query()
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

    def format_data(self):
        df = pd.DataFrame(self.dl_data())
        try:
            df.columns = df.iloc[0]
            df = df.iloc[1:]
        except IndexError:
            print("Index Error, empty df")
        return df

    def get_data(self):
        if self.df is None:
            self.df = self.format_data()
        return self.df

    def explanation(self):
        explanation = "Column Descriptions are as follows:\n"
        for var in self.variables.items():
            var_str = f'{var[0]} is defined as {var[1]["label"]}\n'
            explanation += var_str
        return explanation


class AnalysisChain:
    """Runs basic pandas EDA functions on a dataframe and explains those results
    using GPT 3.5 Turbo
    """

    def __init__(self, df, variables, open_ai_key):
        self.df = df
        self.variables = variables
        self.info = self.df_info()

        template = """
        Role: You are a python output interpreter.
        
        Context:
        You will recieve 4 pieces of information about a pandas dataframe
        The first thing will be a python dictionary where the keys are the column
        names and the values are a description of the column. The second piece of information
        will be the results of calling the describe() method on the dataframe, 
        and the third will be the results of calling the info() method on the dataframe.
        The fourth piece of information is the result of calling value_counts()
        on the dataframe.
        
        Instructions:
        Explain that you ran some python code and are explaining the output of that code.
        
        Read the four pieces of information provided. Explain any key findings
        that you find. Base your analysis solely on the information provided in
        this prompt. Don't make anything up, just provide coherent basic analysis
        based on the information.
        
        -----
        
        Information: {information}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=open_ai_key)
        self.chain = prompt | model | StrOutputParser()

    def df_info(self):
        column_info = {}
        for d in self.variables:
            column_info[d] = self.variables[d]["label"]
        description = self.df.describe()

        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_string = buffer.getvalue()

        value_counts = self.df.value_counts()

        return column_info, description, info_string, value_counts

    def invoke(self):
        return self.chain.invoke({"information": self.info})
