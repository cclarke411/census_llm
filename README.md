# census_llm
This application makes US Census Data easily accessible via natural language queries.

This is a website built for a user to be able to access US Census data and ACS data to get information for any question available in these data sources. This site is meant to return relevant data, perform basic analysis and describe the results of the analysis. The main objective of this site is to make data more accessible for anyone interested in Census or ACS data, whether or not they have a technical background. This site is intended to be a tool that might not find solutions in the first go, but to guide the user through the search for the data. The website talks the user through its result analysis, and the user can accordingly tweak the inputs to get better results.


## Github Repo

[https://github.com/abejburton/census_llm](https://github.com/abejburton/census_llm)

## How to Use It

### Through the Web at [https://censusbot.streamlit.app/](https://censusbot.streamlit.app/)
Enter your Census and OpenAI Keys in the sidebar. Then ask a question in the central text input box!

### Run Locally
1. Clone/download the git repo.
1. Install `requirements.txt` 
1. Create a .env file with `CENSUS_API_KEY` and `OPENAI_API_KEY` as environmental variables.
1. In the command line, type `streamlit run src/main.py` to see the application.

## How it Works
The natural language query is processed using LangChain to customize OpenAI's GPT 3.5 Turbo Model. The query goes through the following steps:

1. **Rephrase** - First the question is slightly rephrased to get a more comprehensive question. (class `RephraseChain`)
1. **Parsing Question Components** - Next, the model identifies the question, relevant geographical area, variables, and datasets in the natural language query. (class `SourceChain`)
1. **Source Identification** - Since the Census offers many different APIs and datasets, we identify the most relevant source to the question using RAG (Retrieval Augmented Generation) (class `SourceRAG`)
1. **Variable Identification** - Variable identification is tricky, as a RAG does not work the best, and there can be many variables (for e.g., the ACS has 60,000+ variables). We take a recursive tree-based approach to finding the best variable. We break the variables into a variable tree on the basis of the different levels of the variable label (e.g. Total > Male > Under 5). (class `VariableTreeChain`)
1. **Geography Identification** - This part of the model identifies the FIPS codes of the geographies mentioned in the question, and breaks it down to the relevant state (and county, if mentioned). If you are using a city, it tries to find the closest state and county (class `GeographyRAG`).  
1. **Census API call construction** - Finally, a Census API call is constructed to get the data based on the most relevant parameters were identified via the RAG. (class `CensusQuery`)
4. **Post Data Analysis** - The retrieved data goes through some simple python analysis which is then fed to GPT 3.5 Turbo and the site displays the GPT interpretation of the python results. (class `AnalysisChain`)

Progress updates are printed out along the way. Sometimes this analysis can take time if the relevant metadata embeddings haven't been downloaded yet or the model is struggling to find relevant results. If the output isn't what you hoped for, try rewording your question or reading more about what Census Data is available in the links provided. The model can make mistakes, and some questions might not fall within the scope of published data.

If results aren't as expected even with a well worded question, try submitting again. LLM's can have inconsistent output so results may vary and trying again might be all it takes for good results. For users inputing your own API key, keep in mind that OpenAI charges on a per token basis.


## Limitations and future goals

1. Running it with the same input can sometimes lead to different outputs.
1. It might not always figure out the exact geographies/variables you ask about, rewording it can help
1. Currently, we have limited it to a smaller list of 10-12 datasets (still hundreds of thousands of variables), we would want to expand this in the future
1. Currently, we have limited the geographic scope to states and counties (or county equvalents), we would want to expand this in the future as well.


## Iterations (or, why this seems heavily overengineered)

Here is a little explanation of why we did what we did.

**Iteration 1**  
Initially, we started with a simple idea - ask GPT3.5 to return the variables it thought would be the best fit for a given question. 

**Iteration 2**  
This was followed by throwing the "Census How To" book at it, along with a RAG that retrieved variables based on the question entered. This showed some early promise but it turned out to be quite false. For example, when we searched for "men under 45", it would pick out "40-44 years" and "45-49 years", but not say "10-14 years".

**Iteration 3**  
This meant we had to not just find similarity with the question but the general theme of the question. We added the `SourceChain` here to split the question into smaller components, and then applied a `VariableRAG`. Again, this showed promise! Taking the previous example further, now we were splitting the question to its core components - "Age and Gender" - and then a found the variables closest to it. Now, we were identifying the variables "under 5 years", "10-14 years", but it had a hard time picking out all 8-9 different variables, more often than not because they were split across multiple documents.

**Iteration 4**  
This led us to asking, how do we split the documents at appropriate points? This was the first version of the tree based approach, where we used a Tree and a RAG at the same time. The biggest problem with this was the absolutely massive number of text-embeddings calls needed. Deeming it far too expensive, we finally figured we did not really need a RAG! A well-designed tree approach worked just as well (if not better).

**Iteration 5**  
This tree approach was based on the variable levels (e.g. Total, Total > Men, Total > Women, Total > Men > Under 5, Total > Men > 6 - 10 etc.). This worked quite well but it was losing the plot once in a while. We fixed this with adding the Census "concept" as the first root. This leads to a larger search space earlier on, but it hones in on the variables quite quickly.

**Iteration 6**  
Once we realised that the breaking up of the question into smaller sections worked better, we broke the rest of the query into smaller sections as well, and used section appropriate search methods

## Feedback
Let us know if you find any bugs and feel free to submit any improvements!

## Work Split

### Combined
Ideation, debugging, cleaning, util functions, prompt engineering, etc.

### Abe Burton
Entirety of the frontend, `AnalysisChain` class, `CensusQuery` class

### Divij Sinha
`VariableTreeChain` and `VarTree` class, `GeographyRAG` class, `SourceRAG` class
