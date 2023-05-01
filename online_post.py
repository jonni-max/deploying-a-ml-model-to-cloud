"""
online_post.py

Run an example POST to the app running online on render.

"""


import pandas as pd

from fastapi.testclient import TestClient

from main import app


def go():
    
    client = TestClient(app)
    
    # Take a record from the census data with salary greater than 50k
    census_data = pd.read_csv('data/census_test.csv')
    input_data = census_data.iloc[9, :]
    
    print(f"\nInput data: \n{input_data.to_json()}")
    
    input_data = input_data.drop(['salary'])
    
    r = client.post(
        "https://mldevops-project-3.onrender.com/predict/",
        data=input_data.to_json()
    )
                         
    print(f"\nOutput from render: \n{r.json()}")    
    print(f"\nStatus code from render: {r.status_code}")
    
    
if __name__ == "__main__":
    
    go()
    
        