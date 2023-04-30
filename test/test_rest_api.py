"""
test_rest_api.py

Tests of the REST API for the POST and GET
"""


import pytest
import json
import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)

def test_api_get():
    
    r = client.get("http://127.0.0.1:8000/")
    
    print(r.json()[0])
    
    assert r.status_code == 200
    assert r.json() == ["Welcome to the udacity-mldevops project 3 salary prediction REST API!"]
    

def test_api_post_gt50k():
    
    # Take a record from the census data with salary greater than 50k
    census_data = pd.read_csv('data/census_clean.csv')
    input_data = census_data[census_data.salary=='<=50K'].drop(['salary'], axis=1).iloc[33, :]
    
    print(f"\nPOST input is:\n{input_data.to_json()}")
    
    r = client.post("http://127.0.0.1:8000/predict/", data=input_data.to_json())
    
    print(f"\nPOST output is:\n{r.json()}")
    
    assert r.status_code == 200
    assert r.json() == {'prediction': ['<=50K']}
    

def test_api_post_lt50k():
    
    # Take a record from the census data with salary less than 50k
    census_data = pd.read_csv('data/census_clean.csv')
    input_data = census_data[census_data.salary=='>50K'].drop(['salary'], axis=1).iloc[33, :]
    
    print(f"\nPOST input is:\n{input_data.to_json()}")
    
    r = client.post("http://127.0.0.1:8000/predict/", data=input_data.to_json())
    
    print(f"\nPOST output is:\n{r.json()}")

    assert r.status_code == 200
    assert r.json() == {'prediction': ['>50K']}
    
    