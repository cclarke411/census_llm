from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv()

### SAMPLES
api_access_url = "http://api.census.gov/data/2020/dec/dp"

variables = {
    "DP1_0028C": {
        "label": "Count!!SEX AND AGE!!Male population!!10 to 14 years",
        "concept": "PROFILE OF GENERAL POPULATION AND HOUSING CHARACTERISTICS",
        "predicateType": "int",
        "group": "DP1",
        "limit": 0,
        "attributes": "DP1_0028CA",
    },
    "DP1_0029C": {
        "label": "Count!!SEX AND AGE!!Male population!!15 to 19 years",
        "concept": "PROFILE OF GENERAL POPULATION AND HOUSING CHARACTERISTICS",
        "predicateType": "int",
        "group": "DP1",
        "limit": 0,
        "attributes": "DP1_0029CA",
    },
    "DP1_0030C": {
        "label": "Count!!SEX AND AGE!!Male population!!20 to 24 years",
        "concept": "PROFILE OF GENERAL POPULATION AND HOUSING CHARACTERISTICS",
        "predicateType": "int",
        "group": "DP1",
        "limit": 0,
        "attributes": "DP1_0030CA",
    },
}

geography_fips = {"state": "49", "county": ["011", "013"]}
###

class Query:
    def __init__(self, api_access_url: str , variables: dict, geography_fips: str):
        self.url = api_access_url
        self.variables = variables
        self.geographies = geography_fips
        
    def build_query(self):
        query = [self.url, '?get=']
        
        # specify dataset variables to extract
        vars = list(self.variables.keys())
        vars_csv = ','.join(vars)
        query.append(vars_csv)
        
        #specify geographies to include
        tract,state,county = ('*','*','*')        
        if 'tract' in self.geographies:
            tract = self.geographies['tract']
            if type(tract) == list:
                tract = ','.join(tract)
        if 'state' in self.geographies:
            state = self.geographies['state']
            if type(state) == list:
                state = ','.join(state)
        if 'county' in self.geographies:
            county = self.geographies['county']
            if type(county) == list:
                county = ','.join(county)
        geo_string = f'&for=tract:{tract}&in=state:{state}&in=county:{county}'

        # include api key at the end of the query
        query.append(geo_string)
        key = os.getenv("CENSUS_API_KEY")
        api_key = f'&key={key}'
        query.append(api_key)
        
        return ''.join(query)
    
    def get_data(self):
        try:
            api_url = self.build_query()
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            
    def format_data(self):
        df = pd.DataFrame(self.get_data())
        df.columns = df.iloc[0]
        df = df[1:]
        return df
            

test = Query(api_access_url, variables, geography_fips)
test.format_data()