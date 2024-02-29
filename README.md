# census_llm
This application makes US Census Data easily accessible via natural language queries.

This is a website that allows for a user to enter a question they would like answered by Census Data. The site returns the relevant data, performs basic analysis, and describes the results of the analysis.

## How to Use It

### Through the Web at {}.com
Enter your Census and OpenAI Keys in the sidebar. Then ask a question in the central text input box!

### Fork this Repo
1. Create a .env file with CENSUS_API_KEY and OPENAI_API_KEY as environmental variables.
2. In the command line, type 'streamlit run src/client/streamlit.py' to see the application.

## How it Works
The natural language query is processed using LangChain to customize OpenAI's GPT 3.5 Turbo Model. The query goes through the following steps:

1. The model identifies the question, relevant geographical area, variables, and datasets in the natural language query.
2. RAG systems based on GPT 3.5 Turbo identify what datasets and variables are most relevant to answering the question.
3. A Census API call is created to get the data based on the most relevant parameters were identified via the RAG.
4. The data goes through some simple python analysis which is displayed along with a GPT based description of the analysis results.
