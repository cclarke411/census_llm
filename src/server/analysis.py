from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
import io

import sys
sys.path.append('/Users/abeburton/Desktop/census_llm/src/')
import server.query_api as q
    
class Analysis:

    def __init__(self, query):
        self.query = query
        self.df = self.query.format_data()
        self.variables = self.query.variables
        self.info = self.df_info()
    
    def df_info(self):
        column_info = {}
        for d in self.variables:
            column_info[d] = self.variables[d]['label']
        description = self.df.describe()
        
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_string = buffer.getvalue()
        
        value_counts = self.df.value_counts()
        
        return column_info, description, info_string, value_counts
    
    def prompt(self):
        template = """You are a python output interpreter.
        
        You will recieve 4 pieces of information about a pandas dataframe
        The first thing will be a python dictionary where the keys are the column
        names and the values are a description of the column. The second piece of information
        will be the results of calling the describe() method on the dataframe, 
        and the third will be the results of calling the info() method on the dataframe.
        The fourth piece of information is the result of calling value_counts()
        on the dataframe.
        
        Read the four pieces of information provided. Explain any key findings
        that you find. Base your analysis solely on the information provided in
        this prompt. Don't make anything up, just provide coherent basic analysis
        based on the information.
        
        -----
        
        Information: {information}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"information": self.info})
    
test = Analysis(query = q.Query(q.api_access_url,q.variables, q.geography_fips))
print(test.prompt())