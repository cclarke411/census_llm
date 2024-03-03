# census_llm
This application makes US Census Data easily accessible via natural language queries.

This is a website that allows for a user to enter a question they would like answered by Census Data. The site returns the relevant data, performs basic analysis, and describes the results of the analysis.
The goal is to make Census/ACS data accessible to anyone interested whether or not they have a technical background. No need to know how to write code, use an API, or even sift through the large Census website. Any user can just create the API Keys, enter them, and ask the website anything in plain english.

## How to Use It

### Through the Web at {}.com
Enter your Census and OpenAI Keys in the sidebar. Then ask a question in the central text input box!

### Fork this Repo
1. Create a .env file with CENSUS_API_KEY and OPENAI_API_KEY as environmental variables.
2. In the command line, type 'streamlit run src/client/streamlit.py' to see the application.

## How it Works
The natural language query is processed using LangChain to customize OpenAI's GPT 3.5 Turbo Model. The query goes through the following steps:

1. The model identifies the question, relevant geographical area, variables, and datasets in the natural language query.
2. RAG (Retrieval Augmented Generation) models based on GPT 3.5 Turbo identify what datasets and variables are most relevant to answering the question.
3. A Census API call is constructed to get the data based on the most relevant parameters were identified via the RAG.
4. The data goes through some simple python analysis which is then fed to GPT 3.5 Turbo and the site displays the GPT interpretation of the python results.

Progress updates are printed out along the way. Sometimes this analysis can take time if the relevant metadata embeddings haven't been downloaded yet or the model is struggling to find relevant results. If the output isn't what you hoped for, try rewording your question or reading more about what Census Data is available in the links provided. The model can make mistakes, and some questions might not fall within the scope of published data.

If results aren't as expected even with a well worded question, try submitting again. LLM's can have inconsistent output so results may vary and trying again might be all it takes for good results. For users inputing your own API key, keep in mind that OpenAI charges on a per token basis.

## Feedback
Let us know if you find any bugs and feel free to submit any improvements!
